import numpy as np
import pandas as pd

def simulate_tumor_advanced(T0: float, r_s: float, r_r: float, K: float, al: float, mu: float, times: np.ndarray, D: bool) -> pd.DataFrame:
    """
    Discrete-time tumor volume simulator that contemplates acquired resistance to treatment.
    It only stores the predicted volume for observed days. 
    
    Input:
    ----------
    T0       : Initial tumor volume (in mm^3)
    r_s      : Tumor growth rate for sensitive to treatment cells
    r_r      : Tumor growth rate for resistant to treatment cells
    K        : Total carrying capacity (in mm^3)
    al       : Drug efficiency (= 0 if no treatment)
    mu       : Mutation rate from sensitive to resistant phenotype
    times    : Array of observation days
    D        : Bool variable indicating presence/absence of treatment 

    Output:
    ----------
    Pandas DataFrame with four columns: 
        "Day"
        "Sensitive_size" : predicted volume of sensitive cells
        "Resistant_size" : predicted volume of resistant cells
        "Size"   : total predicted volume
    """
    max_day = int(times[-1])
    sample_days = set(times)

    T_s: list[float] = [] # Sensitive RTV
    T_r: list[float] = [] # Resistant RTV
    Ts_current = T0
    Tr_current = 0.0

    for i in range(max_day + 1):
        if i in sample_days:
            T_s.append(Ts_current)
            T_r.append(Tr_current)

        growth_s = r_s * Ts_current * (1 - (Ts_current + Tr_current)/K)
        growth_r = r_r * Tr_current * (1 - (Ts_current + Tr_current)/K)

        drug_effect = al * D * Ts_current
        mutation = mu * Ts_current

        # Update values
        Ts_current = max(0.0, Ts_current + (growth_s - drug_effect - mutation) * 1.0)
        Tr_current = max(0.0, Tr_current + (growth_r + mutation) * 1.0)

    return pd.DataFrame({
        "Day": times,
        "Sensitive_size": T_s,
        "Resistant_size": T_r,
        "Size": np.array(T_s) + np.array(T_r)})