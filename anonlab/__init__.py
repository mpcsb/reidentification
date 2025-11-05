# anonlab/__init__.py
from .data import make_synthetic_patients
from .anonymizer import anonymize_qi, collapse_rare_zips, age_bucket, group_size_summary
# forward the helper from the attacker module (attacker.py)
from .attacker import make_attacker_subset_and_validate



__all__ = [
    "make_synthetic_patients",
    "anonymize_qi",
    "collapse_rare_zips",
    "age_bucket",
    "group_size_summary",
    "make_attacker_subset_and_validate",
]
