import numpy as np
import pandas as pd

def simulate_tumor_advanced(T0: float, r_s: float, r_r: float, K: float, al: float, mu: float, times: np.ndarray, D_values: np.ndarray) -> pd.DataFrame:
    """
    Tumor volume simulator for non-uniforn timesteps that contemplates acquired resistance to treatment.

    Input:
    ----------
    T0       : Initial volume (in mm^3)
    r_s      : Tumor growth rate for sensitive to treatment cells
    r_r      : Tumor growth rate for resistant to treatment cells
    K        : Total carrying capacity (in mm^3)
    al       : Drug efficiency (= 0 if no treatment)
    mu       : Mutation rate from sensitive to resistant phenotype
    times    : Array of observation days
    D_values : Array of treatment intensity (dose)

    Output:
    ----------
    Pandas DataFrame with four columns: 
        "DAYS"
        "SENSITIVE" : predicted volume of sensitive cells
        "RESISTANT" : predicted volume of resistant cells
        "VOL"       : total predicted volume
    """
    T_s = np.zeros(len(times)) # Sensitive volume
    T_r = np.zeros(len(times)) # Resistant volume
    T_s[0] = T0

    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        Ts_current, Tr_current = T_s[i-1], T_r[i-1]

        growth_s = r_s * Ts_current * (1 - (Ts_current + Tr_current)/K)
        growth_r = r_r * Tr_current * (1 - (Ts_current + Tr_current)/K)

        drug_effect = al * D_values[i-1] * Ts_current
        mutation = mu * Ts_current

        # Update values
        T_s = max(0, Ts_current + (growth_s - drug_effect - mutation) * dt)
        T_r = max(0, Tr_current + (growth_r + mutation) * dt)

    return pd.DataFrame({
        "DAYS": times,
        "SENSITIVE": T_s,
        "RESISTANT": T_r,
        "VOL": T_s + T_r})