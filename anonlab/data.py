# anonlab/data.py
import numpy as np
import pandas as pd

def make_synthetic_patients(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic toy dataset:
      person_id, age, zip3, sex, lab_glucose
    """
    rng = np.random.default_rng(seed)
    person_id = np.arange(1, n + 1)
    ages = np.arange(18, 91)
    probs = np.linspace(1.0, 2.2, len(ages))
    probs /= probs.sum()
    age = rng.choice(ages, size=n, p=probs)

    zip_choices = np.array([130,130,130,131,131,132,133,134,200,201])
    zip3 = rng.choice(zip_choices, size=n)

    sex = rng.choice(["F","M"], size=n, p=[0.55,0.45])
    lab_glucose = rng.normal(90, 10, size=n).round(1)

    return pd.DataFrame({
        "person_id": person_id,
        "age": age,
        "zip3": zip3,
        "sex": sex,
        "lab_glucose": lab_glucose,
    })
