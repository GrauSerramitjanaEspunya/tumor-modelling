import numpy as np
import pandas as pd

def simulate_tumor_basic(T0: float, r: float, K: float, al: float, times: np.ndarray, D_values: np.ndarray) -> pd.DataFrame:
    """
    Tumor volume simulator for non-uniform timesteps.

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
    T_pred = np.zeros(len(times))
    T_pred[0] = T0

    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        T_current = T_pred[i-1]

        growth = r * T_current * (1 - T_current/K)
        drug_effect = al * D_values[i-1] * T_current

        T_pred[i] = max(0, T_current + (growth - drug_effect)*dt) # To prevent negative volumes

    return pd.DataFrame({
        "DAYS": times,
        "VOL": T_pred})