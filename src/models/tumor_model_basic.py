import numpy as np
import pandas as pd

def simulate_tumor_basic(T0: float, r: float, K: float, al: float, times: np.ndarray, D_values: np.ndarray) -> pd.DataFrame:
    """
    Tumor volume simulator.

    Input:
    ----------
    T0       : Initial volume (in mm^3)
    r        : Tumor growth rate
    K        : Total carrying capacity
    al       : Drug efficiency (= 0 if no treatment)
    times    : Array of observation days
    D_values : Array of treatment intensity (dose)

    Output:
    ----------
    Pandas DataFrame with two columns:
        "DAYS"
        "VOL" : total predicted volume
    """
    dose_lookup = dict(zip(times, D_values))
    max_day = int(times[-1])
    sample_days = set(times)

    T_pred: list[float] = []
    T_current = T0

    for i in range(max_day + 1):
        if i in sample_days:
            T_pred.append(T_current)
            D_current = dose_lookup[i]

        growth = r * T_current * (1 - T_current/K)
        drug_effect = al * D_current * T_current

        T_current = max(0, T_current + (growth - drug_effect) * 1.0) # To prevent negative volumes

    return pd.DataFrame({
        "DAYS": times,
        "VOL": T_pred})