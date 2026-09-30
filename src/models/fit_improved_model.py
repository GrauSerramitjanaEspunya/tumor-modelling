import numpy as np
import pandas as pd
from .improved_model import simulate_tumor
from scipy.optimize import minimize, differential_evolution


def truncate_dataset(df: pd.DataFrame, max_day: int = 15) -> pd.DataFrame:
    """
    Returns a copy of the original dataset containing only samples in which "Day" <= max_day.
    """
    df_truncated = df[df["Day"] <= max_day].copy()
    return df_truncated


def fit_model(data: pd.DataFrame, seed: int, maxiter: int, popsize: int, max_day: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits the basic tumor volume simulator model through differential evolution using the input dataset.

    Output:
    ----------
    Pandas DataFrame with four columns: 
        "Group" : group identificator
        "r"     : estimated growth rate (shared accross groups)
        "K"     : estimated Carrying capacity (shared accross groups)
        "al"    : estimated cellular death rate per group
        "MSE"   : final MSE obtained
    """
    data_clean = truncate_dataset(data, max_day)
    mice_ids = data["ID"].unique()

    bounds = [
        (0.05, 0.8),      # \bar{r} (group average growth rate)
        (3000.0, 7500.0), # K (shared)
        (0.0, 2.0),       # al_1
        (0.0, 2.0),       # al_2
        (0.0, 2.0),       # al_3
        (0.01, 0.30)      # sigma_r (sd of growth rate random)
    ]

    def loss_function(mouse_df: pd.DataFrame, r: float, bar_r: float, sigma_r: float, K: float, al: float) -> float:
        """
        Computes loss value for a single mouse as: log-MSE + Gauss prior penalty
        """
        # Compute predictions
        times = mouse_df["Day"].to_numpy()
        T_obs = mouse_df["Size"].to_numpy()
        T0 = T_obs[0]
        grp = mouse_df["Group"].iloc[0]
        D = grp > 1

        res_df = simulate_tumor(T0, r, K, al, times, D)
        T_pred = res_df["Size"].to_numpy()

        # Compute loss
        mse = float(np.mean((np.log(T_obs) - np.log(T_pred)) ** 2))
        prior = ((r - bar_r) ** 2) / (2 * (sigma_r ** 2))
        return mse + prior

    def objective(params: np.ndarray) -> float:
        """
        Objective function. Runs the simulation for all mice and reports the join MSE + prior penalization.
        """
        bar_r, K, al_1, al_2, al_3, sigma_r = params
        alphas = {1: 0.0, 2: al_1, 3: al_2, 4: al_3} # key = group ; value = alpha value

        total_loss = 0.0
        num_mice = data["ID"].nunique()

        for id in mice_ids:
            mouse_df = data_clean[data_clean["ID"] == id]
            grp = mouse_df["Group"].iloc[0]
            al = alphas[grp]

            # Compute loss -> optimize r_i and compute MSE + prior loss
            res_mouse = minimize(
                fun=lambda r: loss_function(mouse_df, r[0], bar_r, sigma_r, K, al),
                x0=[bar_r],
                bounds=[(0.01, 1.20)],
                method="L-BFGS-B"
            )

            total_loss += res_mouse.fun
            
        return total_loss / num_mice
    
    result = differential_evolution(objective, 
                                    bounds, 
                                    seed=seed,
                                    maxiter=maxiter,
                                    popsize=popsize)


    # Extract optimal params
    opt_bar_r, opt_K, opt_al2, opt_al3, opt_al4, opt_sigma_r = result.x
    opt_alphas = {1: 0.0, 2: opt_al2, 3: opt_al3, 4: opt_al4}

    # Extract optimal mouse results
    mouse_results = []
    for id in mice_ids:
        mouse_df = data_clean[data_clean["ID"] == id]
        grp = mouse_df["Group"].iloc[0]
        al = opt_alphas[grp]

        res_mouse = minimize(
            fun=lambda r: loss_function(mouse_df, r[0], opt_bar_r, opt_sigma_r, opt_K, al),
            x0=[opt_bar_r],
            bounds=[(0.01, 1.20)],
            method="L-BFGS-B"
        )

        mouse_results.append({
            "ID": id,
            "Group": grp,
            "r_id": res_mouse.x[0],
            "eta_i": res_mouse.x[0] - opt_bar_r
        })

    # Define group results
    group_results = pd.DataFrame({
        "Group": [1, 2, 3, 4],
        "bar_r_s": opt_bar_r,
        "K": opt_K,
        "al": [0.0, opt_al2, opt_al3, opt_al4],
        "sigma_r": opt_sigma_r,
        "Global_Loss": result.fun
    })
    
    individual_results = pd.DataFrame(mouse_results)
    return group_results, individual_results