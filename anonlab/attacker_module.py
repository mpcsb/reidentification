# attacker_module.py
"""
Attacker module — one-to-one matching WITHOUT using any sensitive continuous attribute.
Assumption: attacker knows (age, zip3, sex) for victims (attacker_orig).
Works with anonymized dataframe produced by anonymize_qi, which must contain:
    person_id, age_gen, zip3_gen, sex_gen, lab_glucose (lab_glucose ignored here)
Functions:
    build_candidates(attacker_orig, anon_df)
    score_candidates_no_glucose(cands, params...)
    solve_assignment(cands_scored, n_attack, n_anon, time_limit_s=10)
    evaluate_assignments(assignments, attacker_ids, anon_df)
    run_attack(attacker_orig, anon_df, attacker_ids, solver="ortools"|"greedy", params=...)
    sweep_k(...) convenience
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

# try ortools
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False


def build_candidates(attacker_orig: pd.DataFrame, anon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build candidate pairs using only attacker-known QIs: age containment, zip3, sex.
    Returns DataFrame: attacker_idx, anon_idx, zip_ok, sex_ok, width, age_dist
    attacker_orig expected columns: person_id, age, zip3, sex  (order preserved)
    anon_df expected columns: person_id, age_gen (tuple), zip3_gen, sex_gen, ...
    """
    att = attacker_orig.reset_index(drop=True)
    anon = anon_df.reset_index(drop=True)
    rows = []
    for i, arow in att.iterrows():
        for j, orow in anon.iterrows():
            lo, hi = orow["age_gen"]
            age_ok = (hi is None and arow["age"] >= lo) or (hi is not None and lo <= arow["age"] <= hi)
            if not age_ok:
                continue
            zip_ok = (str(arow["zip3"]) == str(orow["zip3_gen"])) or (str(orow["zip3_gen"]) == "other")
            sex_ok = (arow["sex"] == orow["sex_gen"]) or (orow["sex_gen"] == "*")
            # tie-breakers (not sensitive): bucket width and distance to midpoint
            if orow["age_gen"][1] is None:
                width = float(9999)
                midpoint = float(orow["age_gen"][0])
            else:
                width = float(orow["age_gen"][1] - orow["age_gen"][0] + 1)
                midpoint = (orow["age_gen"][0] + orow["age_gen"][1]) / 2.0
            age_dist = float(abs(arow["age"] - midpoint))
            rows.append({
                "attacker_idx": int(i),
                "anon_idx": int(j),
                "zip_ok": bool(zip_ok),
                "sex_ok": bool(sex_ok),
                "width": width,
                "age_dist": age_dist,
            })
    return pd.DataFrame(rows)


def score_candidates_no_glucose(cands: pd.DataFrame,
                                weight_zip: float = 1.0,
                                weight_sex: float = 0.8,
                                age_scale: float = 5.0,
                                width_penalty: float = 0.001) -> pd.DataFrame:
    """
    Score candidate pairs WITHOUT using lab_glucose.
    Returns cands copy with 'score' column.
    - age_sim = 1/(1 + age_dist/age_scale)
    - width_penalty = -(width * width_penalty)
    score = weight_zip*zip_ok + weight_sex*sex_ok + age_sim + width_penalty
    """
    if cands.empty:
        return cands.copy()
    c = cands.copy()
    c["zip_f"] = c["zip_ok"].astype(float)
    c["sex_f"] = c["sex_ok"].astype(float)
    c["age_sim"] = 1.0 / (1.0 + c["age_dist"] / float(age_scale))
    c["width_pen"] = -(c["width"] * float(width_penalty))
    c["score"] = weight_zip * c["zip_f"] + weight_sex * c["sex_f"] + c["age_sim"] + c["width_pen"]
    return c


def _greedy_assignment(cands_scored: pd.DataFrame, n_attack: int, n_anon: int) -> Dict[int, int]:
    """Greedy one-to-one assignment by descending score."""
    assigned_att = set()
    assigned_anon = set()
    assignments = {}
    if cands_scored.empty:
        return assignments
    for r in cands_scored.sort_values("score", ascending=False).itertuples(index=False):
        i = int(r.attacker_idx); j = int(r.anon_idx)
        if i in assigned_att or j in assigned_anon:
            continue
        assignments[i] = j
        assigned_att.add(i); assigned_anon.add(j)
        if len(assigned_att) >= n_attack:
            break
    return assignments


def solve_assignment(cands_scored: pd.DataFrame, n_attack: int, n_anon: int, time_limit_s: int = 10) -> Tuple[Dict[int,int], Optional[float]]:
    """
    Solve max-sum assignment (one-to-one). Uses OR-Tools CBC if available, else raises.
    Returns assignments dict attacker_idx -> anon_idx and objective value (or None).
    """
    if cands_scored.empty:
        return {}, 0.0

    if not ORTOOLS_AVAILABLE:
        raise RuntimeError("ORTools not available in this environment. Use greedy fallback instead.")

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("Failed to create CBC solver via OR-Tools.")

    x = {}
    # create variables for feasible candidate pairs
    for _, row in cands_scored.iterrows():
        i = int(row["attacker_idx"]); j = int(row["anon_idx"])
        x[(i,j)] = solver.IntVar(0, 1, f"x_{i}_{j}")

    # each attacker at most one anon
    for i in range(n_attack):
        solver.Add(sum(x[(ii,jj)] for (ii,jj) in x if ii == i) <= 1)

    # each anon at most one attacker
    for j in range(n_anon):
        solver.Add(sum(x[(ii,jj)] for (ii,jj) in x if jj == j) <= 1)

    # objective
    obj = solver.Objective()
    for _, row in cands_scored.iterrows():
        i = int(row["attacker_idx"]); j = int(row["anon_idx"])
        obj.SetCoefficient(x[(i,j)], float(row["score"]))
    obj.SetMaximization()

    solver.SetTimeLimit(int(time_limit_s * 1000))
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"ORTools solver status: {status}")

    assignments = {}
    for (i,j), var in x.items():
        if var.solution_value() > 0.5:
            assignments[i] = j
    return assignments, float(obj.Value())


def evaluate_assignments(assignments: Dict[int,int], attacker_ids, anon_df: pd.DataFrame) -> Dict:
    """
    Evaluate assignment: compute hits (exact person_id matches).
    attacker_ids: iterable of true person_id values, order matches attacker_orig rows used when building candidates.
    anon_df must be reset_index(drop=True) order matching anon_idx used in candidates.
    Returns dict with hits, hit_rate, and details list.
    """
    anon = anon_df.reset_index(drop=True)
    hits = 0
    details = []
    for i, true_pid in enumerate(attacker_ids):
        assigned = assignments.get(i, None)
        assigned_pid = int(anon.loc[assigned, "person_id"]) if assigned is not None else None
        hit = (assigned_pid == int(true_pid))
        if hit:
            hits += 1
        details.append({"attacker_idx": i, "true_pid": int(true_pid), "assigned_anon_idx": assigned, "assigned_pid": assigned_pid, "hit": bool(hit)})
    return {"n_attack": len(attacker_ids), "hits": hits, "hit_rate": hits / max(1, len(attacker_ids)), "details": details}


def run_attack(attacker_orig: pd.DataFrame, anon_df: pd.DataFrame, attacker_ids,
               *,
               weight_zip: float = 1.0,
               weight_sex: float = 0.8,
               age_scale: float = 5.0,
               width_penalty: float = 0.001,
               solver: str = "ortools",
               time_limit_s: int = 10) -> Dict:
    """
    Run full attack pipeline (build -> score -> solve -> eval).
    solver: "ortools" or "greedy". If "ortools" and not available, raises.
    Returns dict: { assignments, objective, eval, candidates }
    """
    cands = build_candidates(attacker_orig, anon_df)
    if cands.empty:
        return {"assignments": {}, "objective": 0.0, "eval": {"n_attack": len(attacker_ids), "hits": 0, "hit_rate": 0.0, "details": []}, "candidates": cands}

    cands_sc = score_candidates_no_glucose(cands, weight_zip=weight_zip, weight_sex=weight_sex, age_scale=age_scale, width_penalty=width_penalty)
    n_attack = int(attacker_orig.shape[0])
    n_anon = int(anon_df.shape[0])

    if solver == "ortools":
        if not ORTOOLS_AVAILABLE:
            raise RuntimeError("ORTools requested but not available")
        assignments, obj = solve_assignment(cands_sc, n_attack=n_attack, n_anon=n_anon, time_limit_s=time_limit_s)
    elif solver == "greedy":
        assignments = _greedy_assignment(cands_sc, n_attack=n_attack, n_anon=n_anon)
        obj = None
    else:
        raise ValueError("solver must be 'ortools' or 'greedy'")

    evald = evaluate_assignments(assignments, attacker_ids, anon_df)
    return {"assignments": assignments, "objective": obj, "eval": evald, "candidates": cands_sc}


def sweep_k(df, make_anonymize_fn, attacker_frac: float = 0.1, ks=(2,3,5,8,10,15,20), *,
            anonymize_kwargs=None, attack_kwargs=None):
    """
    Convenience: sweep K values and run attack for each K.
    - df: original dataset
    - make_anonymize_fn: callable(df, **anonymize_kwargs) -> anon_df (wrapper around anonymize_qi)
    - anonymize_kwargs: kwargs passed to make_anonymize_fn
    - attack_kwargs: kwargs passed to run_attack
    Returns a DataFrame with summary rows.
    """
    anonymize_kwargs = anonymize_kwargs or {}
    attack_kwargs = attack_kwargs or {}
    rows = []
    for k in ks:
        anonymize_kwargs["k"] = k
        anon_df = make_anonymize_fn(df, **anonymize_kwargs)
        # build attacker sample
        # attacker creation should be external; assume user will create same attacker subset across ks if desired.
        from anonlab import make_attacker_subset_and_validate
        att = make_attacker_subset_and_validate(df, anon_df, fraction=attacker_frac, seed=1)
        attacker_orig = att["attacker_orig"][["person_id","age","zip3","sex"]].reset_index(drop=True)
        attacker_ids = att["attacker_ids"]
        # run greedy attack (fast)
        res = run_attack(attacker_orig, anon_df, attacker_ids, solver=attack_kwargs.get("solver","greedy"), **attack_kwargs)
        rows.append({"k": k, "n_anon": len(anon_df), "n_attack": res["eval"]["n_attack"], "hits": res["eval"]["hits"], "hit_rate": res["eval"]["hit_rate"], "suppressed": anon_df.attrs.get("suppressed", 0)})
    return pd.DataFrame(rows)



# --- attacker helper: build attacker subset + sanity checks (no glucose used here) ---

from typing import Dict
import math
import numpy as np
import pandas as pd

def make_attacker_subset_and_validate(
    orig_df: pd.DataFrame,
    anon_df: pd.DataFrame,
    *,
    fraction: float = 0.1,
    seed: int = 1
) -> Dict[str, object]:
    """
    Sample a subset of victims (by person_id) from the original DF and return:
      - attacker_orig: rows for those person_ids with columns [person_id, age, zip3, sex, lab_glucose]
      - attacker_anon: corresponding anonymized rows (same person_ids) from anon_df
      - attacker_ids: np.ndarray of sampled person_ids (order matches attacker_orig rows)
      - validation: dict with in-bucket containment counts and age-bin width summary
    This DOES NOT use lab_glucose for any matching logic; it's kept in attacker_orig for completeness.
    """
    rng = np.random.RandomState(seed)
    n = int(len(orig_df))
    sample_size = max(1, int(math.ceil(n * float(fraction))))
    sampled = rng.choice(orig_df["person_id"].values, size=sample_size, replace=False)

    attacker_orig = (
        orig_df.set_index("person_id")
               .loc[sampled]
               .reset_index()
               [["person_id", "age", "zip3", "sex", "lab_glucose"]]
    )
    attacker_ids = attacker_orig["person_id"].to_numpy()

    # map attacker persons to their buckets in the released data
    attacker_anon = anon_df[anon_df["person_id"].isin(attacker_ids)].reset_index(drop=True)

    # containment sanity (original age ∈ anonymized age_gen)
    def _age_in_bucket(age: int, g) -> bool:
        lo, hi = g
        return (hi is None and age >= lo) or (hi is not None and lo <= age <= hi)

    merged = (
        attacker_orig.set_index("person_id")[["age"]]
        .merge(attacker_anon.set_index("person_id")[["age_gen"]], left_index=True, right_index=True, how="left")
    )

    mismatches = int((~merged.apply(lambda r: _age_in_bucket(int(r["age"]), r["age_gen"]), axis=1)).sum())
    in_bucket_count = int(len(merged) - mismatches)

    # widths summary for non-topcoded buckets (diagnostic)
    widths = []
    for g in attacker_anon["age_gen"]:
        if isinstance(g, tuple) and g[1] is not None:
            widths.append(g[1] - g[0] + 1)
    widths_summary = pd.Series(widths).describe().to_dict() if widths else {}

    return {
        "attacker_orig": attacker_orig.reset_index(drop=True),
        "attacker_anon": attacker_anon,
        "attacker_ids": attacker_ids,
        "validation": {
            "attacker_sample_size": int(sample_size),
            "in_bucket_count": in_bucket_count,
            "mismatches": mismatches,
            "widths_summary": widths_summary,
        },
    }
