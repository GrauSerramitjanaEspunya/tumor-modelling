import numpy as np
import pandas as pd
from .tumor_model_basic import simulate_tumor_basic
from scipy.optimize import differential_evolution


def loss_function(T_obs: np.ndarray, T_pred: np.ndarray) -> float:
    """
    Returns the MSE between the observed log RTV and the predicted log RTV of the tumor.
    """
    return float(np.mean((np.log(T_obs) - np.log(T_pred)) ** 2))


def fit_model_basic(data: pd.DataFrame, seed: int, maxiter: int, popsize: int) -> pd.DataFrame:
    """
    Fits the basic tumor volume simulator model through differential evolution using the entire dataset.

    Output:
    ----------
    Pandas DataFrame with four columns: 
        "Group" : group identificator
        "r"     : estimated growth rate (shared accross groups)
        "K"     : estimated Carrying capacity (shared accross groups)
        "al"    : estimated cellular death rate per group
        "MSE"   : final MSE obtained
    """
    bounds = [
        (0.05, 0.8),      # r (shared)
        (4000.0, 7500.0), # K (shared)
        (0.0, 2.0),       # al_1
        (0.0, 2.0),       # al_2
        (0.0, 2.0)        # al_3
    ]

    mice_ids = data["ID"].unique()

    def objective(params: np.ndarray) -> float:
        """
        Objective function. Runs the simulation for all mice and reports the join MSE.
        """
        r, K = params[0], params[1]
        alphas = [0.0, params[2], params[3], params[4]]

        total_loss = 0.
        num_mice = data["ID"].nunique()

        for id in mice_ids:
            T_obs = data.loc[data["ID"] == id, "Size"].to_numpy()
            times = data.loc[data["ID"] == id, "Day"].to_numpy()
            T0 = T_obs[0]

            grp = data.loc[data["ID"] == id, "Group"].iloc[0]
            D = grp > 1

            sim_df = simulate_tumor_basic(
                T0,
                r,
                K,
                alphas[grp-1],
                times,
                D
            )

            T_pred = sim_df["Size"].to_numpy()

            total_loss += loss_function(T_obs, T_pred)

        return total_loss / num_mice
    
    result = differential_evolution(objective, 
                                    bounds, 
                                    seed=seed,
                                    maxiter=maxiter,
                                    popsize=popsize)

    opt_params = result.x
    df_results = pd.DataFrame({
        "Group": [1, 2, 3, 4],
        "r": opt_params[0],
        "K": opt_params[1],
        "al": [0.0, opt_params[2], opt_params[3], opt_params[4]],
        "MSE": result.fun
    })

    return df_results
