"""
app.py — Flask backend for the KG Mutation Testing research dashboard.

Reuses the notebook's own implementation:
  - kg_testing_lib.py      : recommender, mutation operators M1-M5, behavioral metrics
                              (verbatim copy of the notebook's Section 7B module)
  - behavioral_analysis.py : BR1-BR5 classification + anomaly scoring
                              (verbatim logic from notebook Sections 18/19)

All runtime data (product/KG maps, preference profiles, similarity tables, frozen
development-calibrated thresholds, frozen percentile lookups, frozen anomaly cutoff)
comes from demo_bundle.pkl, which is exported directly from the live notebook run
(see NOTEBOOK_EXPORT_CELL.py). Nothing here invents data, metrics, or thresholds.
"""
import os
import pickle
import random
import importlib

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

import kg_testing_lib as lib
import behavioral_analysis as ba

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(APP_DIR, "demo_bundle.pkl")

app = Flask(__name__)


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

BUNDLE = None


def load_bundle():
    global BUNDLE
    with open(BUNDLE_PATH, "rb") as f:
        BUNDLE = pickle.load(f)

    importlib.reload(lib)
    lib.init(
        product_attrs_base=BUNDLE["product_attrs_base"],
        product_category_map=BUNDLE["product_category_map"],
        product_brand_map=BUNDLE["product_brand_map"],
        product_department_map=BUNDLE["product_department_map"],
        valid_cat_dept_pairs=BUNDLE["valid_cat_dept_pairs"],
        valid_brand_cat_dept_triples=BUNDLE["valid_brand_cat_dept_triples"],
        categories_by_department=BUNDLE["categories_by_department"],
        brands_by_cat_dept=BUNDLE["brands_by_cat_dept"],
        pref_c=BUNDLE["pref_c"], pref_b=BUNDLE["pref_b"], pref_d=BUNDLE["pref_d"],
        user_purchases=BUNDLE["user_purchases"],
        category_sim=BUNDLE["category_sim"], brand_sim=BUNDLE["brand_sim"],
        baseline_weights=BUNDLE["baseline_weights"],
        top_k=BUNDLE["top_k"],
        mutation_distance=BUNDLE["mutation_distance"],
    )
    ba.init(
        rc_thresh=BUNDLE["RC_THRESH"], ss_thresh=BUNDLE["SS_THRESH"],
        ds_thresh=BUNDLE["DS_THRESH"], loc_thresh=BUNDLE["LOC_THRESH"],
        dev_low_rel_cut=BUNDLE["DEV_LOW_REL_CUT"], dev_low_loc_cut=BUNDLE["DEV_LOW_LOC_CUT"],
        pctl_rc=BUNDLE["PCTL_RC"], pctl_ss=BUNDLE["PCTL_SS"],
        pctl_ds=BUNDLE["PCTL_DS"], pctl_loc=BUNDLE["PCTL_LOC"],
        anomaly_score_cutoff=BUNDLE["ANOMALY_SCORE_CUTOFF"],
        anomaly_weights=BUNDLE["ANOMALY_WEIGHTS"],
    )


load_bundle()

MUTATION_NAMES = {
    "M1": "Category Reassignment",
    "M2": "Brand Reassignment",
    "M3": "Category Relation Removal",
    "M4": "Brand Relation Removal",
    "M5": "Category + Brand Reassignment",
}
BR_NAMES = {
    "BR1": "Bounded Sensitivity",
    "BR2": "Semantic Consistency",
    "BR3": "Locality of Impact",
    "BR4": "Irrelevant Mutation Stability",
    "BR5": "Diversity Preservation",
}


def product_card(pid):
    info = BUNDLE["product_lookup"].get(int(pid), {})
    return {
        "product_id": int(pid),
        "name": info.get("name", f"Product {pid}"),
        "category": info.get("category"),
        "brand": info.get("brand"),
        "department": info.get("department"),
        "price": info.get("retail_price"),
    }


def top10_diff(orig, mutated):
    orig_rank = {p: i + 1 for i, p in enumerate(orig)}
    mut_rank = {p: i + 1 for i, p in enumerate(mutated)}
    rows = []
    for p in sorted(set(orig) | set(mutated), key=lambda p: (mut_rank.get(p, 999), orig_rank.get(p, 999))):
        in_o, in_m = p in orig_rank, p in mut_rank
        if in_o and in_m:
            status = "unchanged" if orig_rank[p] == mut_rank[p] else ("moved_up" if mut_rank[p] < orig_rank[p] else "moved_down")
        elif in_m and not in_o:
            status = "new"
        else:
            status = "disappeared"
        rows.append({
            **product_card(p),
            "orig_rank": orig_rank.get(p),
            "mutated_rank": mut_rank.get(p),
            "status": status,
        })
    rows.sort(key=lambda r: (r["mutated_rank"] is None, r["mutated_rank"] or 0, r["orig_rank"] or 0))
    return rows


@app.route("/api/meta")
def meta():
    return jsonify({
        "mutation_types": [{"code": c, "name": MUTATION_NAMES[c]} for c in BUNDLE["mutation_types"]],
        "br_names": BR_NAMES,
        "anomaly_cutoff": BUNDLE["ANOMALY_SCORE_CUTOFF"],
        "anomaly_weights": BUNDLE["ANOMALY_WEIGHTS"],
        "top_k": BUNDLE["top_k"],
        "baseline_weights": BUNDLE["baseline_weights"],
        "pipeline_stages": ["Original KG", "Mutation", "Mutated KG",
                             "Recommendation Comparison", "Behavioural Testing", "Anomaly Detection"],
        "demo_note": BUNDLE.get("demo_note", ""),
    })


@app.route("/api/users")
def users():
    out = []
    for uid in BUNDLE["demo_users"]:
        purchases = BUNDLE["user_purchases"].get(uid, set())
        cats = sorted({BUNDLE["product_category_map"].get(p) for p in purchases if BUNDLE["product_category_map"].get(p)})
        out.append({
            "user_id": int(uid),
            "n_purchases": len(purchases),
            "n_categories": len(cats),
            "sample_categories": cats[:4],
        })
    return jsonify(out)


def deterministic_rng(user_id, mtype):
    # Fixed per (user, mutation_type) seed so a candidate list and a subsequent run-test
    # request for the same (user, product, mutation_type) reproduce the identical, already-
    # validated mutation deterministically (the notebook's own apply_mutation() is itself
    # randomized only in which VALID option it picks when several exist, e.g. M2's brand).
    return random.Random(1234 + user_id * 7919 + (hash(mtype) % 100000))


@app.route("/api/users/<int:user_id>/candidates")
def candidates(user_id):
    mtype = request.args.get("mutation_type", "M1")
    pool = lib.get_candidate_pool(user_id, top_n=200)
    rng = deterministic_rng(user_id, mtype)
    valid = []
    for pid in pool:
        mutated_attrs = lib.apply_mutation(pid, mtype, rng)
        if mutated_attrs is not None and lib.validate_mutation(pid, mtype, mutated_attrs):
            rel = lib.mutation_relevance(user_id, mtype, pid, mutated_attrs)
            card = product_card(pid)
            valid.append({
                **card,
                "mutated_category": mutated_attrs["category"],
                "mutated_brand": mutated_attrs["brand"],
                "mutated_department": mutated_attrs["department"],
                "user_relevance": round(float(rel), 4),
            })
        if len(valid) >= 25:
            break
    valid.sort(key=lambda r: -r["user_relevance"])
    return jsonify(valid)


@app.route("/api/run-test", methods=["POST"])
def run_test():
    payload = request.get_json(force=True)
    uid = int(payload["user_id"])
    mtype = payload["mutation_type"]
    pid = int(payload["product_id"])

    # Re-derive the mutation the same deterministic way /candidates listed it, so the
    # product picked in the UI maps to exactly the same mutated attrs shown there.
    rng = deterministic_rng(uid, mtype)
    mutated_attrs = None
    pool = lib.get_candidate_pool(uid, top_n=200)
    for cand_pid in pool:
        ma = lib.apply_mutation(cand_pid, mtype, rng)
        if cand_pid == pid and ma is not None and lib.validate_mutation(cand_pid, mtype, ma):
            mutated_attrs = ma
            break
    if mutated_attrs is None:
        return jsonify({"error": "No valid mutation could be generated for this (user, product, mutation_type)."}), 400

    row = {
        "mutation_id": f"DEMO-{uid}-{pid}-{mtype}",
        "user_id": uid, "product_id": pid, "mutation_type": mtype,
        "mutation_distance": BUNDLE["mutation_distance"][mtype],
        "user_relevance": lib.mutation_relevance(uid, mtype, pid, mutated_attrs),
        "original_category": lib.PRODUCT_CATEGORY_MAP.get(pid),
        "mutated_category": mutated_attrs["category"],
        "original_brand": lib.PRODUCT_BRAND_MAP.get(pid),
        "mutated_brand": mutated_attrs["brand"],
        "original_department": lib.PRODUCT_DEPARTMENT_MAP.get(pid),
        "mutated_department": mutated_attrs["department"],
    }

    result = lib.execute_mutation_row(row)
    df = pd.DataFrame([result])
    df = ba.apply_behavioral_relations(df)
    df = ba.add_anomaly_scores(df)
    r = df.iloc[0]

    anomaly_score = float(r["anomaly_score_statistical"])
    cutoff = float(BUNDLE["ANOMALY_SCORE_CUTOFF"])
    verdict = "ANOMALOUS" if anomaly_score > cutoff else "NON-ANOMALOUS"

    brs_fired = [k for k in ["BR1", "BR2", "BR3", "BR4", "BR5"] if bool(r[f"{k}_violation"])]

    response = {
        "mutation_id": row["mutation_id"],
        "user": {"user_id": uid},
        "mutation_type": mtype,
        "mutation_name": MUTATION_NAMES[mtype],
        "target_product": product_card(pid),
        "kg_change": {
            "original": {"category": row["original_category"], "brand": row["original_brand"], "department": row["original_department"]},
            "mutated": {"category": row["mutated_category"] if pd.notna(row["mutated_category"]) else None,
                        "brand": row["mutated_brand"] if pd.notna(row["mutated_brand"]) else None,
                        "department": row["mutated_department"]},
        },
        "top10": {
            "original": [product_card(p) for p in result["orig_top10"]],
            "mutated": [product_card(p) for p in result["mutated_top10"]],
            "diff": top10_diff(result["orig_top10"], result["mutated_top10"]),
        },
        "metrics": {
            "ranking_change": round(float(r["ranking_change"]), 4),
            "semantic_shift": round(float(r["semantic_shift"]), 4),
            "brand_shift": round(float(r["brand_shift"]), 4),
            "diversity_shift": round(float(r["diversity_shift"]), 4),
            "diversity_decreased": bool(r["diversity_decreased"]),
            "locality": round(float(r["locality"]), 4),
            "locality_applicable": bool(r["locality_applicable"]),
            "user_relevance": round(float(r["user_relevance"]), 4),
            "mutation_distance": int(r["mutation_distance"]),
            "product_score_delta": None if pd.isna(r["product_score_delta"]) else round(float(r["product_score_delta"]), 4),
            "ranking_change_level": r["ranking_change_level"],
        },
        "behavioral_relations": {
            k: {"name": BR_NAMES[k], "violated": bool(r[f"{k}_violation"])} for k in ["BR1", "BR2", "BR3", "BR4", "BR5"]
        },
        "anomaly": {
            "statistical_score": round(anomaly_score, 4),
            "weighted_score": round(float(r["anomaly_score_weighted"]), 4),
            "frozen_cutoff": round(cutoff, 4),
            "verdict": verdict,
            "n_violations": int(r["n_violations"]),
        },
        "interpretation": build_interpretation(mtype, r, brs_fired, verdict, cutoff, anomaly_score),
    }
    return jsonify(response)


def build_interpretation(mtype, r, brs_fired, verdict, cutoff, score):
    parts = []
    parts.append(
        f"{MUTATION_NAMES[mtype]} ({mtype}) produced a ranking_change (RBO-based) of "
        f"{r['ranking_change']:.3f}, classified as {r['ranking_change_level']} for this mutation type."
    )
    parts.append(
        f"Semantic shift (category-distribution JS-divergence) = {r['semantic_shift']:.3f}; "
        f"diversity shift (entropy delta) = {r['diversity_shift']:.3f} "
        f"({'decreased' if r['diversity_decreased'] else 'did not decrease'} diversity)."
    )
    if r["locality_applicable"]:
        parts.append(f"Locality of the change = {r['locality']:.3f} (new Top-10 items were compared to the affected category/brand).")
    else:
        parts.append("Locality is not applicable here: the mutation did not introduce any new item into the Top-10.")
    if brs_fired:
        parts.append("Behavioral relations flagged: " + ", ".join(f"{k} ({BR_NAMES[k]})" for k in brs_fired) + ".")
    else:
        parts.append("No behavioral relation (BR1-BR5) was violated.")
    parts.append(
        f"Statistical anomaly score = {score:.3f} vs the frozen, development-calibrated cutoff "
        f"({cutoff:.3f}, 90th percentile of development anomaly scores) -> {verdict}."
    )
    if verdict == "ANOMALOUS":
        parts.append("This flags the mutation as warranting investigation — it is not a confirmed software bug.")
    return " ".join(parts)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
