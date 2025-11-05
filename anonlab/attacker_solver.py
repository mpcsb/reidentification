# attacker_solver.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict
try:
    from ortools.linear_solver import pywraplp
except Exception as e:
    raise RuntimeError("ortools not available. Install with: pip install ortools") from e

def build_candidates(attacker_orig: pd.DataFrame, anon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with columns: attacker_idx, anon_idx, age_ok, zip_ok, sex_ok,
    glucose_diff.
    Indices are integer positions (0..n-1) for stable solver indexing.
    """
    # prepare arrays
    att = attacker_orig.reset_index(drop=True)
    anon = anon_df.reset_index(drop=True)
    rows = []
    for i, arow in att.iterrows():
        for j, orow in anon.iterrows():
            # age containment check
            lo, hi = orow["age_gen"]
            age_ok = (hi is None and arow["age"] >= lo) or (hi is not None and lo <= arow["age"] <= hi)
            if not age_ok:
                continue
            # zip compatibility: either exact match or anon says "other"
            zip_ok = (str(arow["zip3"]) == str(orow["zip3_gen"])) or (str(orow["zip3_gen"]) == "other")
            # sex compatibility: either matches or anon is "*"
            sex_ok = (arow["sex"] == orow["sex_gen"]) or (orow["sex_gen"] == "*")
            # glucose diff (absolute)
            glucose_diff = abs(float(arow["lab_glucose"]) - float(orow["lab_glucose"]))
            rows.append({
                "attacker_idx": int(i),
                "anon_idx": int(j),
                "age_ok": age_ok,
                "zip_ok": bool(zip_ok),
                "sex_ok": bool(sex_ok),
                "glucose_diff": glucose_diff,
            })
    return pd.DataFrame(rows)

def score_candidates(cands: pd.DataFrame, weight_zip=1.0, weight_sex=0.8, glucose_scale: float = 10.0) -> pd.DataFrame:
    """
    Add a 'score' column to candidates:
      score = weight_zip*(zip_ok) + weight_sex*(sex_ok) + (1/(1+glucose_diff/glucose_scale))
    Higher is better.
    """
    if cands.empty:
        return cands
    c = cands.copy()
    c["zip_ok_f"] = c["zip_ok"].astype(float)
    c["sex_ok_f"] = c["sex_ok"].astype(float)
    c["glucose_sim"] = 1.0 / (1.0 + c["glucose_diff"] / float(glucose_scale))
    c["score"] = weight_zip * c["zip_ok_f"] + weight_sex * c["sex_ok_f"] + c["glucose_sim"]
    return c

def solve_assignment(
    cands_scored: pd.DataFrame,
    n_attack: int,
    n_anon: int,
    time_limit_s: int = 10
) -> Tuple[Dict[int,int], float]:
    """
    Solve max-sum assignment using OR-Tools CBC solver.
    Returns:
      assignments: dict attacker_idx -> anon_idx
      objective_value (float)
    """
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver not available in ortools build.")

    x = {}
    # create variables
    for _, row in cands_scored.iterrows():
        i = int(row["attacker_idx"])
        j = int(row["anon_idx"])
        x[(i,j)] = solver.IntVar(0, 1, f"x_{i}_{j}")

    # each attacker assigned to at most 1 anon
    for i in range(n_attack):
        solver.Add(sum(x[(i,j)] for (ii,j) in x if ii == i) <= 1)

    # each anon assigned to at most 1 attacker
    for j in range(n_anon):
        solver.Add(sum(x[(i,jj)] for (i,jj) in x if jj == j) <= 1)

    # objective
    objective = solver.Objective()
    for _, row in cands_scored.iterrows():
        i = int(row["attacker_idx"]); j = int(row["anon_idx"])
        objective.SetCoefficient(x[(i,j)], float(row["score"]))
    objective.SetMaximization()

    solver.SetTimeLimit(int(time_limit_s * 1000))
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"Solver status: {status}")

    assignments = {}
    for (i,j), var in x.items():
        if var.solution_value() > 0.5:
            assignments[i] = j
    obj_val = objective.Value()
    return assignments, float(obj_val)

def evaluate_assignments(assignments: Dict[int,int], attacker_orig: pd.DataFrame, anon_df: pd.DataFrame, attacker_ids: np.ndarray) -> Dict:
    """
    Compute Hit@1 (exact ID recovered) and return a small diagnostics dict.
    attacker_orig must be reset_indexed (attacker_idx -> row).
    attacker_ids: array of original person_ids for the attacker sample (length == n_attack)
    """
    n_attack = len(attacker_ids)
    hits = 0
    details = []
    # anon_df indexed by integer position with original person_id in person_id column
    anon = anon_df.reset_index(drop=True)
    for i, true_pid in enumerate(attacker_ids):
        assigned = assignments.get(i, None)
        hit = False
        assigned_pid = None
        if assigned is not None:
            assigned_pid = int(anon.loc[assigned, "person_id"])
            hit = (assigned_pid == int(true_pid))
        if hit:
            hits += 1
        details.append({"attacker_idx": i, "true_pid": int(true_pid), "assigned_anon_idx": assigned, "assigned_pid": assigned_pid, "hit": hit})
    return {"n_attack": n_attack, "hits": hits, "hit_rate": hits / max(1, n_attack), "details": details}

# convenience top-level function
def run_one_to_one_attack(
    attacker_orig: pd.DataFrame,
    anon_df: pd.DataFrame,
    attacker_ids: np.ndarray,
    *,
    weight_zip=1.0, weight_sex=0.8, glucose_scale=10.0,
    time_limit_s: int = 10
) -> Dict:
    """
    attacker_orig: DataFrame of the victims known to attacker (columns: person_id, age, zip3, sex, lab_glucose)
    anon_df: anonymized DF (columns: person_id, age_gen, zip3_gen, sex_gen, lab_glucose)
    attacker_ids: array of person_id values (same order as attacker_orig)
    Returns dict with assignments, objective, evaluation.
    """
    # ensure attacker_orig is in same order as attacker_ids
    attacker_orig = attacker_orig.reset_index(drop=True)
    anon_df = anon_df.reset_index(drop=True)

    cands = build_candidates(attacker_orig, anon_df)
    if cands.empty:
        return {"assignments": {}, "objective": 0.0, "eval": {"n_attack": len(attacker_ids), "hits": 0, "hit_rate": 0.0}, "candidates": cands}

    cands_scored = score_candidates(cands, weight_zip=weight_zip, weight_sex=weight_sex, glucose_scale=glucose_scale)
    n_attack = attacker_orig.shape[0]
    n_anon = anon_df.shape[0]
    assignments, obj = solve_assignment(cands_scored, n_attack=n_attack, n_anon=n_anon, time_limit_s=time_limit_s)
    evald = evaluate_assignments(assignments, attacker_orig, anon_df, attacker_ids)
    return {"assignments": assignments, "objective": obj, "eval": evald, "candidates": cands_scored}
