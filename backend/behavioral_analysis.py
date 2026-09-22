"""
behavioral_analysis.py

Behavioral-relation classification (BR1-BR5) and anomaly scoring, extracted VERBATIM
from the research notebook (Sections 18, 18A, 19, 19A: cells defining robust_z,
calibrate_thresholds, classify_level, apply_behavioral_relations, build_percentile_lookup,
pctl_rank, statistical_anomaly_score, weighted_anomaly_score, add_anomaly_scores).

No metric, threshold, or scoring logic has been changed. The only addition is init(),
which injects the frozen, development-calibrated objects (RC_THRESH, SS_THRESH, DS_THRESH,
LOC_THRESH, DEV_LOW_REL_CUT, DEV_LOW_LOC_CUT, PCTL_RC/SS/DS/LOC, ANOMALY_SCORE_CUTOFF,
CONFIG["ANOMALY_WEIGHTS"]) that the notebook computed once from the development sample and
then reused unchanged for held-out evaluation -- mirroring the injection pattern already
used by kg_testing_lib.init().
"""
import numpy as np
import pandas as pd

# ---- Globals populated by init() (frozen development calibration) ----
RC_THRESH = None
SS_THRESH = None
DS_THRESH = None
LOC_THRESH = None
DEV_LOW_REL_CUT = None
DEV_LOW_LOC_CUT = None
PCTL_RC = None
PCTL_SS = None
PCTL_DS = None
PCTL_LOC = None
ANOMALY_SCORE_CUTOFF = None
ANOMALY_WEIGHTS = None


def init(rc_thresh, ss_thresh, ds_thresh, loc_thresh, dev_low_rel_cut, dev_low_loc_cut,
         pctl_rc, pctl_ss, pctl_ds, pctl_loc, anomaly_score_cutoff, anomaly_weights):
    global RC_THRESH, SS_THRESH, DS_THRESH, LOC_THRESH, DEV_LOW_REL_CUT, DEV_LOW_LOC_CUT
    global PCTL_RC, PCTL_SS, PCTL_DS, PCTL_LOC, ANOMALY_SCORE_CUTOFF, ANOMALY_WEIGHTS
    RC_THRESH = rc_thresh
    SS_THRESH = ss_thresh
    DS_THRESH = ds_thresh
    LOC_THRESH = loc_thresh
    DEV_LOW_REL_CUT = dev_low_rel_cut
    DEV_LOW_LOC_CUT = dev_low_loc_cut
    PCTL_RC = pctl_rc
    PCTL_SS = pctl_ss
    PCTL_DS = pctl_ds
    PCTL_LOC = pctl_loc
    ANOMALY_SCORE_CUTOFF = anomaly_score_cutoff
    ANOMALY_WEIGHTS = anomaly_weights


# ============================================================
# Section 18 / 18A — Behavioral Relations (verbatim)
# ============================================================

def robust_z(x, median, mad):
    denom = 1.4826 * mad if mad > 1e-9 else 1e-9
    return (x - median) / denom


def classify_level(value, thresh_row):
    z = robust_z(value, thresh_row["median"], thresh_row["mad"])
    if value <= thresh_row["p75"] and abs(z) < 2:
        return "EXPECTED"
    if value <= thresh_row["p95"] and abs(z) < 3.5:
        return "BORDERLINE"
    return "ANOMALOUS"


def apply_behavioral_relations(
    results_df: pd.DataFrame,
    rc_thresh=None, ss_thresh=None, ds_thresh=None, loc_thresh=None,
    low_rel_cut: dict = None, low_loc_cut: dict = None,
) -> pd.DataFrame:
    rc_thresh = RC_THRESH if rc_thresh is None else rc_thresh
    ss_thresh = SS_THRESH if ss_thresh is None else ss_thresh
    ds_thresh = DS_THRESH if ds_thresh is None else ds_thresh
    loc_thresh = LOC_THRESH if loc_thresh is None else loc_thresh
    low_rel_cut = DEV_LOW_REL_CUT if low_rel_cut is None else low_rel_cut
    low_loc_cut = DEV_LOW_LOC_CUT if low_loc_cut is None else low_loc_cut

    df = results_df.copy()

    br1, br2, br3, br4, br5, rc_levels = [], [], [], [], [], []
    for _, r in df.iterrows():
        mt = r["mutation_type"]
        rel_cut = low_rel_cut[mt]
        loc_cut = low_loc_cut.get(mt, 0.0)

        rc_level = classify_level(r["ranking_change"], rc_thresh.loc[mt])
        ss_level = classify_level(r["semantic_shift"], ss_thresh.loc[mt])
        ds_level = classify_level(r["diversity_shift"], ds_thresh.loc[mt])

        loc_applicable = bool(r["locality_applicable"])
        loc_is_low = loc_applicable and (r["locality"] < loc_cut)

        br1_violation = (r["mutation_distance"] == 1) and (r["user_relevance"] < rel_cut) and (rc_level == "ANOMALOUS")
        br2_violation = loc_applicable and (ss_level == "ANOMALOUS") and loc_is_low
        br3_violation = loc_applicable and loc_is_low and (rc_level in ("BORDERLINE", "ANOMALOUS"))
        br4_violation = (r["user_relevance"] < rel_cut) and (rc_level == "ANOMALOUS")
        br5_violation = (r["user_relevance"] < rel_cut) and r["diversity_decreased"] and (ds_level == "ANOMALOUS")

        br1.append(br1_violation); br2.append(br2_violation); br3.append(br3_violation)
        br4.append(br4_violation); br5.append(br5_violation); rc_levels.append(rc_level)

    df["BR1_violation"] = br1; df["BR2_violation"] = br2; df["BR3_violation"] = br3
    df["BR4_violation"] = br4; df["BR5_violation"] = br5
    df["ranking_change_level"] = rc_levels
    df["n_violations"] = df[["BR1_violation", "BR2_violation", "BR3_violation", "BR4_violation", "BR5_violation"]].sum(axis=1)
    return df


# ============================================================
# Section 19 — Anomaly Scoring (verbatim)
# ============================================================

def pctl_rank(value, sorted_arr):
    if len(sorted_arr) == 0:
        return 0.5
    return float(np.searchsorted(sorted_arr, value, side="right") / len(sorted_arr))


def statistical_anomaly_score(row) -> float:
    mt = row["mutation_type"]
    r_rc = pctl_rank(row["ranking_change"], PCTL_RC.get(mt, np.array([])))
    r_ss = pctl_rank(row["semantic_shift"], PCTL_SS.get(mt, np.array([])))
    r_ds = pctl_rank(row["diversity_shift"], PCTL_DS.get(mt, np.array([])))

    components = [r_rc, r_ss, r_ds]
    if row["locality_applicable"]:
        r_loc = 1 - pctl_rank(row["locality"], PCTL_LOC.get(mt, np.array([])))
        components.append(r_loc)

    base = float(np.mean(components))
    violation_component = min(row["n_violations"], 1) * 0.10
    return float(min(base + violation_component, 1.0))


def weighted_anomaly_score(row, weights=None) -> float:
    if weights is None:
        weights = ANOMALY_WEIGHTS
    locality_term = (1 - row["locality"]) if row["locality_applicable"] else 0.0
    violation_component = min(row["n_violations"], 1)
    return float(
        weights["ranking"] * row["ranking_change"] +
        weights["semantic"] * row["semantic_shift"] +
        weights["diversity"] * min(row["diversity_shift"] / 2.0, 1.0) +
        weights["locality"] * locality_term +
        weights["violation"] * violation_component
    )


def add_anomaly_scores(df, weights=None):
    df = df.copy()
    df["anomaly_score_statistical"] = df.apply(statistical_anomaly_score, axis=1)
    df["anomaly_score_weighted"] = df.apply(lambda r: weighted_anomaly_score(r, weights), axis=1)
    return df
