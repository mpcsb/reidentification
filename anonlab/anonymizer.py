# anonlab/anonymizer.py
from typing import Optional, Tuple
import numpy as np
import pandas as pd

def collapse_rare_zips(df: pd.DataFrame, zip_col: str = "zip3", min_frac: float = 0.05) -> pd.DataFrame:
    counts = df[zip_col].value_counts(normalize=True)
    rare = counts[counts < min_frac].index
    out = df.copy()
    out["zip3_collapsed"] = out[zip_col].astype(str).where(~out[zip_col].isin(rare), other="other")
    return out


def age_bucket(age: int, width: int, level: int,
               topcode_start: Optional[int] = None,
               min_age: int = 0) -> Tuple[int, Optional[int]]:
    """
    Correct topcode semantics:
      - If *age* >= topcode_start -> return (topcode_start, None)
      - Else compute bucket [lo, hi]
      - If bucket starts >= topcode_start -> topcode bucket
      - If bucket overlaps threshold -> clip hi = topcode_start - 1
    """
    if topcode_start is not None and age >= topcode_start:
        return (topcode_start, None)

    w = max(1, width * (2 ** level))
    lo = min_age + ((int(age) - min_age) // w) * w
    hi = lo + w - 1

    if topcode_start is not None:
        if lo >= topcode_start:
            return (topcode_start, None)
        if hi >= topcode_start:
            hi = topcode_start - 1

    return (int(lo), int(hi))


def anonymize_qi(
    df: pd.DataFrame,
    *,
    k: int = 5,
    age_bin_width: int = 5,
    topcode_start: Optional[int] = 75,
    rare_zip_min_frac: float = 0.05,
    max_iter: int = 8,
    extra_iter: int = 6,
    allow_zip_force: bool = True,
    allow_sex_any: bool = True,
    allow_suppression: bool = True,   # ← NEW: default ON
) -> pd.DataFrame:
    """
    Per-row widening with targeted fallback; optionally suppress infeasible stragglers.
    Output columns: person_id, age_gen, zip3_gen, sex_gen, lab_glucose
    """
    df = df.copy().reset_index(drop=True)
    df = collapse_rare_zips(df, zip_col="zip3", min_frac=rare_zip_min_frac)
    df["_level"] = 0
    df["sex_gen"] = df["sex"].astype(str)
    df["zip3_gen"] = df["zip3_collapsed"].astype(str)

    def recompute(df_):
        df_ = df_.copy()
        df_["age_gen"] = [
            age_bucket(a, age_bin_width, lvl, topcode_start)
            for a, lvl in zip(df_["age"], df_["_level"])
        ]
        return df_

    df = recompute(df)

    # Primary widening
    for _ in range(max_iter):
        df["_gsize"] = df.groupby(["age_gen","zip3_gen","sex_gen"])["person_id"].transform("size")
        if (df["_gsize"] >= k).all():
            break
        mask = df["_gsize"] < k
        df.loc[mask, "_level"] += 1
        df = recompute(df)

    # Targeted fallback cycles
    for _ in range(extra_iter):
        df["_gsize"] = df.groupby(["age_gen","zip3_gen","sex_gen"])["person_id"].transform("size")
        mask = df["_gsize"] < k
        if not mask.any():
            break

        if allow_zip_force:
            df.loc[mask, "zip3_gen"] = "other"

        df["_gsize"] = df.groupby(["age_gen","zip3_gen","sex_gen"])["person_id"].transform("size")
        mask = df["_gsize"] < k
        if not mask.any():
            break

        if allow_sex_any:
            df.loc[mask, "sex_gen"] = "*"

        df["_gsize"] = df.groupby(["age_gen","zip3_gen","sex_gen"])["person_id"].transform("size")
        mask = df["_gsize"] < k
        if not mask.any():
            break

        df.loc[mask, "_level"] += 1
        df = recompute(df)

    out = df[["person_id","age_gen","zip3_gen","sex_gen","lab_glucose"]].copy()

    # Final suppression (e.g., tiny topcoded bucket)
    if allow_suppression:
        sizes = out.groupby(["age_gen","zip3_gen","sex_gen"])["person_id"].transform("size")
        kept = sizes >= k
        suppressed = int((~kept).sum())
        if suppressed:
            out = out.loc[kept].reset_index(drop=True)
            out.attrs["suppressed"] = suppressed
        else:
            out.attrs["suppressed"] = 0

    return out.reset_index(drop=True)





def group_size_summary(anon_df: pd.DataFrame):
    s = anon_df.groupby(["age_gen", "zip3_gen", "sex_gen"])["person_id"].size()
    return s.describe().to_dict()
