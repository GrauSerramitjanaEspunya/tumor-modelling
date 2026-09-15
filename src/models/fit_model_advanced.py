import numpy as np
import pandas as pd
from tumor_model_advanced import simulate_tumor_advanced
from scipy.optimize import differential_evolution


def loss_function(T_obs: np.ndarray, T_pred: np.ndarray) -> float:
    """
    Returns the MSE between the observed log volume and the predicted log volume of the tumor.
    """
    return float(np.mean((T_obs - T_pred) ** 2))


def fit_model(T_obs: np.ndarray, D_values: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    """
    Fits a model accross 5 mice (1 control + 4 treated), sharing r_s (growth rate for sensible cells), r_r (growth rate for resitant cells), 
    mu (mutation rate) and K (carrying capacity)
    """
    bounds = [
        (0.0, 2.0),  # r_s (shared)
        (0.0, 2.0),  # r_r (shared)
        (5.0, 7.6),  # K (shared)
        (0.0, 0.05), # al_1
        (0.0, 0.05), # al_2
        (0.0, 0.05), # al_3
        (0.0, 0.05), # al_4
        (0.0, 0.05)  # mu (shared)
    ]

    T_obs.reshape(5, 11) # each line represents a mouse, each column a day
    D_values.reshape(5, 11)

    def objective(params: np.ndarray) -> float:
        """
        Objective function. Runs the simulation for all mice and reports the join MSE.
        """
        r_s, r_r, K, mu = params[0], params[1], params[2], params[7]
        alphas = [0.0, params[3], params[4], params[5], params[6]]

        num_mice = len(T_obs)
        total_loss = 0.

        for i in range(num_mice):
            try:
                T0_i = T_obs[i, 0]
                sim_df = simulate_tumor_advanced(
                    T0_i,
                    r_s,
                    r_r,
                    K,
                    alphas[i],
                    mu,
                    times,
                    D_values[i]
                )

                T_pred = sim_df["VOL"].values()

                total_loss += loss_function(T_obs[i], T_pred)
            except:
                return 1e6

        return total_loss / num_mice
    
    result = differential_evolution(objective, bounds, seed=23)

    opt_params = result.x
    df_results = pd.DataFrame({
        "ID": [i for i in range(len(T_obs))],
        "r": [opt_params[0] for _ in range(len(T_obs))],
        "K": [opt_params[1] for _ in range(len(T_obs))],
        "al": [0.0, opt_params[2], opt_params[3], opt_params[4], opt_params[5]],
        "MSE": [result.fun for _ in range(len(T_obs))]
    })

    return df_results