import numpy as np
import pandas as pd

def simulate_tumor_basic(T0: float, r: float, K: float, al: float, times: np.ndarray, D: bool) -> pd.DataFrame:
    """
    Discrete-time tumor volume simulator. It only stores the predicted volume for observed days.

    Input:
    ----------
    T0       : Initial tumor volume (in mm^3)
    r        : Tumor growth rate
    K        : Total carrying capacity (in mm^3)
    al       : Drug efficiency (= 0 if no treatment)
    times    : Array of observation days
    D        : Bool variable indicating presence/absence of treatment 

    Output:
    ----------
    Pandas DataFrame with two columns:
        "Day"
        "Size" : total predicted tumor size (in mm^3)
    """
    max_day = int(times[-1])
    sample_days = set(times)

    T_pred: list[float] = []
    T_current = T0

    for i in range(max_day + 1):
        if i in sample_days:
            T_pred.append(T_current)

        growth = r * T_current * (1 - T_current/K)
        drug_effect = al * D * T_current

        T_current = max(0, T_current + (growth - drug_effect) * 1.0) # To prevent negative volumes

    return pd.DataFrame({
        "Day": times,
        "Size": T_pred})