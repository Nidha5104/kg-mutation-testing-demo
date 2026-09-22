"""
kg_testing_lib.py

Core "system under test" for the KG-mutation behavioral-testing experiment:
recommender scoring, the five mutation operators (M1-M5), mutation validity checks,
mutation relevance, RBO ranking-change, semantic-shift / diversity metrics, semantic
locality, and the end-to-end per-mutation execution function.

This lives in a REAL file on disk (not inline in the notebook) specifically so that
coverage.py can attribute genuine, file-and-line-level statement/branch coverage to it.
All large runtime data (product attribute maps, user preference profiles, similarity
tables, config) is injected via init() rather than hard-coded, since it depends on the
BigQuery data pulled at notebook run time.
"""
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as shannon_entropy

# ---- Globals populated by init() ----
PRODUCT_ATTRS_BASE = None          # DataFrame indexed by product_id: [category, brand, department]
PRODUCT_CATEGORY_MAP = None        # dict product_id -> category
PRODUCT_BRAND_MAP = None           # dict product_id -> brand
PRODUCT_DEPARTMENT_MAP = None      # dict product_id -> department
VALID_CAT_DEPT_PAIRS = None        # set of (category, department)
VALID_BRAND_CAT_DEPT_TRIPLES = None  # set of (brand, category, department)
CATEGORIES_BY_DEPARTMENT = None    # dict department -> set(category)
BRANDS_BY_CAT_DEPT = None          # dict (category, department) -> set(brand)
PREF_C = None                      # dict user_id -> {category: pref}
PREF_B = None                      # dict user_id -> {brand: pref}
PREF_D = None                      # dict user_id -> {department: pref}
USER_PURCHASES = None              # dict user_id -> set(product_id)  (NEVER mutated)
CATEGORY_SIM = None                # dict (cat1, cat2) -> jaccard sim
BRAND_SIM = None                   # dict (brand1, brand2) -> jaccard sim
BASELINE_WEIGHTS = None            # dict wC/wB/wD
TOP_K = 10
MUTATION_DISTANCE = None           # dict mutation_type -> int


def init(product_attrs_base, product_category_map, product_brand_map, product_department_map,
         valid_cat_dept_pairs, valid_brand_cat_dept_triples, categories_by_department,
         brands_by_cat_dept, pref_c, pref_b, pref_d, user_purchases,
         category_sim, brand_sim, baseline_weights, top_k, mutation_distance):
    """Inject all runtime data computed in the notebook from the live BigQuery pull."""
    global PRODUCT_ATTRS_BASE, PRODUCT_CATEGORY_MAP, PRODUCT_BRAND_MAP, PRODUCT_DEPARTMENT_MAP
    global VALID_CAT_DEPT_PAIRS, VALID_BRAND_CAT_DEPT_TRIPLES, CATEGORIES_BY_DEPARTMENT, BRANDS_BY_CAT_DEPT
    global PREF_C, PREF_B, PREF_D, USER_PURCHASES, CATEGORY_SIM, BRAND_SIM
    global BASELINE_WEIGHTS, TOP_K, MUTATION_DISTANCE
    PRODUCT_ATTRS_BASE = product_attrs_base
    PRODUCT_CATEGORY_MAP = product_category_map
    PRODUCT_BRAND_MAP = product_brand_map
    PRODUCT_DEPARTMENT_MAP = product_department_map
    VALID_CAT_DEPT_PAIRS = valid_cat_dept_pairs
    VALID_BRAND_CAT_DEPT_TRIPLES = valid_brand_cat_dept_triples
    CATEGORIES_BY_DEPARTMENT = categories_by_department
    BRANDS_BY_CAT_DEPT = brands_by_cat_dept
    PREF_C = pref_c
    PREF_B = pref_b
    PREF_D = pref_d
    USER_PURCHASES = user_purchases
    CATEGORY_SIM = category_sim
    BRAND_SIM = brand_sim
    BASELINE_WEIGHTS = baseline_weights
    TOP_K = top_k
    MUTATION_DISTANCE = mutation_distance


# ============================================================
# RECOMMENDER
# ============================================================

def score_candidates_for_user(user_id, product_attrs, purchased_products, weights=None):
    """Vectorized scoring of all NON-PURCHASED candidate products for one user."""
    if weights is None:
        weights = BASELINE_WEIGHTS
    wC, wB, wD = weights["wC"], weights["wB"], weights["wD"]

    pc = PREF_C.get(user_id, {})
    pb = PREF_B.get(user_id, {})
    pd_ = PREF_D.get(user_id, {})

    cand = product_attrs[~product_attrs.index.isin(purchased_products)].copy()
    cand["score"] = (
        wC * cand["category"].map(pc).fillna(0.0) +
        wB * cand["brand"].map(pb).fillna(0.0) +
        wD * cand["department"].map(pd_).fillna(0.0)
    )
    return cand


def top_k_recommendations(user_id, product_attrs, purchased_products, k=None, weights=None):
    """Top-K product_ids, deterministic tie-break by ascending product_id."""
    if k is None:
        k = TOP_K
    scored = score_candidates_for_user(user_id, product_attrs, purchased_products, weights)
    scored = scored.reset_index().rename(columns={"index": "product_id"})
    scored = scored.sort_values(["score", "product_id"], ascending=[False, True])
    return scored["product_id"].head(k).tolist()


def get_candidate_pool(user_id, top_n=200, weights=None):
    """
    Non-purchased products a user could plausibly be recommended, ranked by their CURRENT
    (pre-mutation) score. Mutation targets are sampled from this pool (see apply_mutation
    callers) rather than from the user's own purchased products, so mutations can actually
    reach the recommendable candidate set and have a chance of affecting the Top-K.
    Restricting to the top-N current candidates (rather than the full ~29k catalogue) is a
    documented, defensible sampling choice: mutating a product with near-zero score for this
    user could never move into their Top-10 regardless of the mutation, so it would not be a
    meaningful test case. This choice is made BEFORE any mutation or behavioral result is
    computed, so it does not introduce selection bias toward large post-mutation changes.
    """
    purchased = USER_PURCHASES.get(user_id, set())
    scored = score_candidates_for_user(user_id, PRODUCT_ATTRS_BASE, purchased, weights)
    scored = scored.reset_index().rename(columns={"index": "product_id"})
    scored = scored.sort_values(["score", "product_id"], ascending=[False, True])
    return scored["product_id"].head(top_n).tolist()


# ============================================================
# MUTATION OPERATORS (M1-M5)
# ============================================================

def apply_mutation(product_id, mutation_type, rng):
    """
    Returns a dict {category, brand, department} describing the SINGLE product's new state,
    or None if no domain-valid mutation could be generated for this (product, type) pair
    (caller should resample a different candidate product rather than inventing an invalid one).
    """
    orig_cat = PRODUCT_CATEGORY_MAP.get(product_id)
    orig_brand = PRODUCT_BRAND_MAP.get(product_id)
    orig_dept = PRODUCT_DEPARTMENT_MAP.get(product_id)
    if pd.isna(orig_cat) or pd.isna(orig_brand) or pd.isna(orig_dept):
        return None

    if mutation_type == "M1":
        # FIX (Revision 3): the target category must be one that has actually been observed
        # WITH THIS PRODUCT'S OWN BRAND in this department - not merely "any category observed
        # in this department" (which could pair the unchanged brand with a category it has never
        # been observed with, e.g. inventing a (Nike, Dresses, Men) combination that never
        # existed in the data). This mirrors the M2/M5 validity rule and matches the project's
        # "observed valid configuration" requirement.
        candidates = sorted({
            c for (b, c, d) in VALID_BRAND_CAT_DEPT_TRIPLES
            if b == orig_brand and d == orig_dept and c != orig_cat
        })
        if not candidates:
            return None
        new_cat = rng.choice(candidates)
        return {"category": new_cat, "brand": orig_brand, "department": orig_dept}

    if mutation_type == "M2":
        candidates = sorted(BRANDS_BY_CAT_DEPT.get((orig_cat, orig_dept), set()) - {orig_brand})
        if not candidates:
            return None
        new_brand = rng.choice(candidates)
        return {"category": orig_cat, "brand": new_brand, "department": orig_dept}

    if mutation_type == "M3":
        return {"category": None, "brand": orig_brand, "department": orig_dept}

    if mutation_type == "M4":
        return {"category": orig_cat, "brand": None, "department": orig_dept}

    if mutation_type == "M5":
        # FIX (Revision 3): both category AND brand must actually change (mutation_distance=2
        # means TWO attributes altered - the previous implementation only guaranteed a different
        # (brand, category) PAIR, which could be satisfied by changing just one of the two,
        # e.g. same category + different brand). The resulting (new_brand, new_category,
        # department) triple must also be an observed combination.
        candidates = [
            (b, c) for (b, c, d) in VALID_BRAND_CAT_DEPT_TRIPLES
            if d == orig_dept and c != orig_cat and b != orig_brand
        ]
        if not candidates:
            return None
        new_brand, new_cat = rng.choice(candidates)
        return {"category": new_cat, "brand": new_brand, "department": orig_dept}

    raise ValueError(f"Unknown mutation type {mutation_type}")


def validate_mutation(product_id, mutation_type, mutated_attrs):
    """Domain-validity check, applied BEFORE the mutation is accepted."""
    if mutated_attrs is None:
        return False
    cat, brand, dept = mutated_attrs["category"], mutated_attrs["brand"], mutated_attrs["department"]

    if mutation_type == "M1":
        # Must be an observed (brand, category, department) triple, category must actually
        # differ, and brand/department must be exactly unchanged.
        return (
            (brand, cat, dept) in VALID_BRAND_CAT_DEPT_TRIPLES
            and cat != PRODUCT_CATEGORY_MAP.get(product_id)
            and brand == PRODUCT_BRAND_MAP.get(product_id)
            and dept == PRODUCT_DEPARTMENT_MAP.get(product_id)
        )
    if mutation_type == "M2":
        return (brand, cat, dept) in VALID_BRAND_CAT_DEPT_TRIPLES and cat == PRODUCT_CATEGORY_MAP.get(product_id)
    if mutation_type == "M3":
        # FIX: use pd.isna() rather than "is None". A removed attribute is represented as Python
        # None when apply_mutation() builds it in memory, but if this function is instead called
        # with a value that has round-tripped through a pandas DataFrame (e.g. a re-validation
        # audit reading mutations_df), pandas commonly coerces that None to NaN. "cat is None"
        # would then incorrectly return False for a correctly-removed category. pd.isna() treats
        # both representations as "removed" so validation is robust regardless of the caller.
        return pd.isna(cat) and brand == PRODUCT_BRAND_MAP.get(product_id)
    if mutation_type == "M4":
        return pd.isna(brand) and cat == PRODUCT_CATEGORY_MAP.get(product_id)
    if mutation_type == "M5":
        # Both category AND brand must differ from the original, department must be unchanged,
        # and the resulting triple must be an observed combination.
        return (
            (brand, cat, dept) in VALID_BRAND_CAT_DEPT_TRIPLES
            and cat != PRODUCT_CATEGORY_MAP.get(product_id)
            and brand != PRODUCT_BRAND_MAP.get(product_id)
            and dept == PRODUCT_DEPARTMENT_MAP.get(product_id)
        )
    return False


def affected_attributes(mutation_type, original_category, mutated_category, original_brand, mutated_brand):
    """
    Which category/brand values are actually part of the mutated region, restricted to the
    attribute(s) THIS operator touches. E.g. M1 only touches category, so brand must not be
    included even though it is (unchanged) present on the row - otherwise locality would be
    inflated by coincidental brand overlap that the mutation never altered.

    Uses pd.notna() rather than "is not None": a removed attribute (M3/M4) may arrive here as
    either Python None (built in-memory by apply_mutation) or NaN (if the caller's row came from
    a pandas DataFrame, which commonly coerces None to NaN) - pd.notna() filters out both
    consistently so a stray NaN never gets treated as a real category/brand value.
    """
    affected_categories, affected_brands = set(), set()
    if mutation_type in ("M1", "M3", "M5"):
        affected_categories = {c for c in [original_category, mutated_category] if pd.notna(c)}
    if mutation_type in ("M2", "M4", "M5"):
        affected_brands = {b for b in [original_brand, mutated_brand] if pd.notna(b)}
    return affected_categories, affected_brands


def mutation_relevance(user_id, mutation_type, product_id, mutated_attrs):
    pc = PREF_C.get(user_id, {})
    pb = PREF_B.get(user_id, {})
    orig_cat = PRODUCT_CATEGORY_MAP.get(product_id)
    orig_brand = PRODUCT_BRAND_MAP.get(product_id)

    if mutation_type == "M1":
        return (pc.get(orig_cat, 0.0) + pc.get(mutated_attrs["category"], 0.0)) / 2
    if mutation_type == "M2":
        return (pb.get(orig_brand, 0.0) + pb.get(mutated_attrs["brand"], 0.0)) / 2
    if mutation_type == "M3":
        return pc.get(orig_cat, 0.0)
    if mutation_type == "M4":
        return pb.get(orig_brand, 0.0)
    if mutation_type == "M5":
        rel_c = (pc.get(orig_cat, 0.0) + pc.get(mutated_attrs["category"], 0.0)) / 2
        rel_b = (pb.get(orig_brand, 0.0) + pb.get(mutated_attrs["brand"], 0.0)) / 2
        return (rel_c + rel_b) / 2
    raise ValueError(mutation_type)


# ============================================================
# BEHAVIORAL METRICS
# ============================================================

def rbo(list1, list2, p=0.9):
    """
    Finite-list Rank-Biased Overlap (Webber, Moffat & Zobel, 2010), extrapolated version
    for two rankings of (possibly unequal, here equal) finite depth k.

    FIX (Revision 3): the previous implementation only summed the weighted-overlap term and
    omitted the extrapolation term p^k * (overlap_at_k / k). That is mathematically NOT RBO -
    it under-counts agreement and, critically, does not return 1.0 for identical finite lists
    (e.g. two identical Top-10 lists previously produced RBO=0.6513 / ranking_change=0.3487
    instead of RBO=1.0 / ranking_change=0.0). This silently inflated ranking_change for every
    mutation in the notebook, including ones that changed nothing. See the RBO sanity test
    immediately after this cell, which is now part of the pipeline gate.

    Returns:
        1.0 for identical rankings (up to depth k)
        0.0 for completely disjoint rankings
    """
    if not list1 and not list2:
        return 1.0

    k = max(len(list1), len(list2))
    if k == 0:
        return 1.0

    seen1, seen2 = set(), set()
    weighted_overlap = 0.0

    for d in range(1, k + 1):
        if d <= len(list1):
            seen1.add(list1[d - 1])
        if d <= len(list2):
            seen2.add(list2[d - 1])

        overlap = len(seen1 & seen2)
        agreement = overlap / d
        weighted_overlap += (1 - p) * (p ** (d - 1)) * agreement

    final_overlap = len(seen1 & seen2) / k

    return float(weighted_overlap + (p ** k) * final_overlap)


def ranking_change(orig_top10, mutated_top10, p=0.9):
    return 1.0 - rbo(orig_top10, mutated_top10, p)


def distribution_shift(orig_list, mutated_list, attr_map):
    orig_vals = [attr_map.get(p) for p in orig_list]
    mut_vals = [attr_map.get(p) for p in mutated_list]
    universe = sorted(set(v for v in orig_vals + mut_vals if v is not None))
    if not universe:
        return 0.0

    def _dist(vals):
        c = pd.Series(vals).value_counts()
        return np.array([c.get(u, 0) for u in universe], dtype=float) / max(len(vals), 1)

    p_orig, p_mut = _dist(orig_vals), _dist(mut_vals)
    dist = jensenshannon(p_orig, p_mut, base=2)
    return float(dist ** 2) if not np.isnan(dist) else 0.0


def semantic_shift(orig_list, mutated_list):
    return distribution_shift(orig_list, mutated_list, PRODUCT_CATEGORY_MAP)


def brand_shift(orig_list, mutated_list):
    return distribution_shift(orig_list, mutated_list, PRODUCT_BRAND_MAP)


def list_entropy(item_list, attr_map):
    vals = [attr_map.get(p) for p in item_list if attr_map.get(p) is not None]
    if not vals:
        return 0.0
    counts = pd.Series(vals).value_counts(normalize=True).values
    return float(shannon_entropy(counts))


def diversity_shift(orig_list, mutated_list):
    h_orig = list_entropy(orig_list, PRODUCT_CATEGORY_MAP)
    h_mut = list_entropy(mutated_list, PRODUCT_CATEGORY_MAP)
    return abs(h_orig - h_mut), (h_mut < h_orig), h_orig, h_mut


def locality_score(affected_categories, affected_brands, orig_top10, mutated_top10):
    """
    Locality in [0,1], restricted to items that are NEW in the mutated Top-10 (additions,
    which capture both "the mutated product itself moved in" and "some other product was
    pulled in by the ranking shift"). Returns (score, applicable):
      - applicable=False, score=1.0 when the mutated Top-10 introduces no new items at all
        (nothing to localize - this is a distinct, documented convention, not silently mixed
        into "genuinely local" cases; downstream code excludes non-applicable rows from
        locality-based calibration/statistics).
    """
    changed = set(mutated_top10) - set(orig_top10)
    if not changed:
        return 1.0, False
    scores = []
    for p in changed:
        p_cat = PRODUCT_CATEGORY_MAP.get(p)
        p_brand = PRODUCT_BRAND_MAP.get(p)
        cat_sims = [CATEGORY_SIM.get((p_cat, ac), 0.0) for ac in affected_categories] if affected_categories else [0.0]
        brand_sims = [BRAND_SIM.get((p_brand, ab), 0.0) for ab in affected_brands] if affected_brands else [0.0]
        scores.append(max(max(cat_sims), max(brand_sims)))
    return float(np.mean(scores)), True


# ============================================================
# END-TO-END EXECUTION
# ============================================================

def execute_mutation_row(row, weights=None):
    uid, pid, mtype = row["user_id"], row["product_id"], row["mutation_type"]
    purchased = USER_PURCHASES.get(uid, set())

    orig_top10 = top_k_recommendations(uid, PRODUCT_ATTRS_BASE, purchased, weights=weights)

    mutated_attrs_table = PRODUCT_ATTRS_BASE.copy()
    mutated_attrs_table.loc[pid, ["category", "brand", "department"]] = [
        row["mutated_category"], row["mutated_brand"], row["mutated_department"]
    ]
    mutated_top10 = top_k_recommendations(uid, mutated_attrs_table, purchased, weights=weights)

    rc = ranking_change(orig_top10, mutated_top10)
    ss = semantic_shift(orig_top10, mutated_top10)
    bs = brand_shift(orig_top10, mutated_top10)
    dshift, dec, h_o, h_m = diversity_shift(orig_top10, mutated_top10)

    affected_cats, affected_brands = affected_attributes(
        mtype, row["original_category"], row["mutated_category"], row["original_brand"], row["mutated_brand"]
    )
    loc, loc_applicable = locality_score(affected_cats, affected_brands, orig_top10, mutated_top10)

    # score delta on the mutated product itself, for the sanity-test / pre-flight audit
    orig_score_row = score_candidates_for_user(uid, PRODUCT_ATTRS_BASE, purchased - {pid}, weights)
    mut_score_row = score_candidates_for_user(uid, mutated_attrs_table, purchased - {pid}, weights)
    orig_score = float(orig_score_row.loc[pid, "score"]) if pid in orig_score_row.index else np.nan
    mut_score = float(mut_score_row.loc[pid, "score"]) if pid in mut_score_row.index else np.nan
    score_delta = (mut_score - orig_score) if (not np.isnan(orig_score) and not np.isnan(mut_score)) else np.nan

    return {
        "mutation_id": row["mutation_id"], "user_id": uid, "product_id": pid, "mutation_type": mtype,
        "mutation_distance": row["mutation_distance"], "user_relevance": row["user_relevance"],
        "original_category": row["original_category"], "mutated_category": row["mutated_category"],
        "original_brand": row["original_brand"], "mutated_brand": row["mutated_brand"],
        "original_department": row["original_department"], "mutated_department": row["mutated_department"],
        "orig_top10": orig_top10, "mutated_top10": mutated_top10,
        "ranking_change": rc, "semantic_shift": ss, "brand_shift": bs,
        "diversity_shift": dshift, "diversity_decreased": dec,
        "entropy_orig": h_o, "entropy_mutated": h_m,
        "locality": loc, "locality_applicable": loc_applicable,
        "product_score_delta": score_delta,
    }
