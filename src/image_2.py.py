import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as opt
from sklearn.decomposition import PCA

def neg_log_likelihood(params, y):
    """
    Compute the negative log-likelihood for an AR(1) model.

    Model: y[t] = c + a * y[t-1] + ε[t], where ε[t] ~ N(0, σ²)
    
    Inputs:
        - params: list [c, a, log_sigma], where:
            - c: AR intercept
            - a: AR coefficient
            - log_sigma: log of standard deviation (ensures positivity)
        - y: 1D array of latent data.

    Output:
        - Negative log-likelihood value (scalar)
    """
    c, a, log_sigma = params
    sigma = np.exp(log_sigma)  # Ensure positivity of standard deviation
    residuals = y[1:] - (c + a * y[:-1])  # Compute residuals
    n = len(residuals)
    # Negative log-likelihood function
    nll = n * np.log(sigma) + 0.5 * n * np.log(2 * np.pi) + np.sum(residuals**2) / (2 * sigma**2)
    return nll

def mle_ar1(y):
    """
    Compute Maximum Likelihood Estimates (MLE) for AR(1) parameters.

    Inputs:
        - y: 1D array of latent data.

    Outputs:
        - c_hat: Estimated intercept
        - a_hat: Estimated AR coefficient
    """
    # Initial parameter estimates using OLS
    c0 = np.mean(y[1:] - y[:-1])  # Mean residual for initial c estimate
    a0 = np.corrcoef(y[:-1], y[1:])[0, 1]  # Correlation coefficient as initial a estimate
    sigma0 = np.std(y[1:] - (c0 + a0 * y[:-1]))  # Estimate residual variance

    # Set initial parameters and constraints
    init = np.array([c0, a0, np.log(sigma0 + 1e-6)])  # Start with log_sigma for positivity
    bounds = [(None, None), (-1, 1), (None, None)]  # a must be between -1 and 1

    # Optimize to find MLE
    res = opt.minimize(lambda params: neg_log_likelihood(params, y), init, bounds=bounds)

    return res.x[:2] if res.success else (np.nan, np.nan)  # Return estimates if optimization succeeded

class IMAGEModelMatrix:
    def __init__(self, num_variables):
        """
        Initialize the IMAGE model.

        Inputs:
            - num_variables: Number of variables (e.g., Tmax, Tmin, Precip, etc.)

        Attributes:
            - monthly_param_avg: Monthly estimated AR(1) parameters
            - residuals: Residuals stored per month
        """
        self.num_variables = num_variables
        self.months = 12
        self.monthly_param_avg = None  # Will store (12, num_variables, 2) for (c, a) values
        self.residuals = {m: [] for m in range(1, 13)}  # Store residuals by month

    def normal_quantile_transform(self, X):
        """
        Transform observed data to latent Gaussian space using NQT.

        Inputs:
            - X: 1D array of observed data
        
        Output:
            - Latent Gaussian transformed values
        """
        ranks = stats.rankdata(X, method='average') / (X.size + 1)
        return stats.norm.ppf(ranks)

    def fit(self, X, dates):
        """
        Estimate AR(1) parameters month-by-month using **only consecutive** data points.

        Inputs:
            - X: Array of shape (T, num_variables) containing observed data.
            - dates: List of datetime objects of length T.

        Outputs:
            - Updates self.monthly_param_avg with estimated (c, a) values.
            - Stores residuals for each month.
        """
        T, S = X.shape
        monthly_params = np.zeros((12, S, 2))  # To store c, a per month and variable

        for m in range(1, 13):  # Loop over 12 months
            month_pairs = {s: [] for s in range(S)}  # Dictionary to collect (y_t-1, y_t) pairs

            # Collect only adjacent values for AR(1) estimation
            for t in range(1, T):
                if dates[t].month == m and dates[t - 1].month == m:
                    for s in range(S):
                        month_pairs[s].append((X[t - 1, s], X[t, s]))

            # Estimate AR(1) parameters for each variable
            for s in range(S):
                if len(month_pairs[s]) < 2:
                    monthly_params[m - 1, s, :] = np.nan
                    continue

                y_prev, y_curr = zip(*month_pairs[s])
                y_prev, y_curr = np.array(y_prev), np.array(y_curr)
                y_latent = self.normal_quantile_transform(y_curr)

                c_hat, a_hat = mle_ar1(y_latent)
                monthly_params[m - 1, s, 0] = c_hat
                monthly_params[m - 1, s, 1] = a_hat

                # Compute residuals and store
                residuals = y_latent[1:] - (c_hat + a_hat * y_latent[:-1])
                self.residuals[m].extend(residuals)

        self.monthly_param_avg = monthly_params  # Save fitted parameters

    def simulate_residuals(self, month):
        """
        Resample residuals using EOF-based resampling.

        Inputs:
            - month: The month (1-12) to sample residuals from.

        Outputs:
            - Simulated residual vector for that month.
        """
        S = self.num_variables
        E = np.array(self.residuals[month]).reshape(-1, S)  # Convert residuals into matrix

        # EOF Decomposition (SVD)
        U, Sigma, VT = np.linalg.svd(E, full_matrices=False)
        G = U * Sigma  # Principal component time series

        # Resample principal components
        n_modes = min(S, G.shape[1])
        P = np.zeros(n_modes)
        for i in range(n_modes):
            r = np.random.randint(0, G.shape[0])
            P[i] = G[r, i]

        # Reconstruct new residuals
        R = (VT.T @ P).reshape(1, S)
        return R

    def simulate(self, n_years, initial_condition):
        """
        Simulate daily time series using AR(1) process with EOF-resampled residuals.

        Inputs:
            - n_years: Number of years to simulate.
            - initial_condition: Initial state of the variables.

        Outputs:
            - Y_latent: Simulated latent time series.
        """
        n_days = n_years * 365
        S = self.num_variables
        Y_latent = np.zeros((n_days, S))
        Y_latent[0, :] = initial_condition  # Set initial condition

        for t in range(1, n_days):
            m = (t // 30) % 12  # Month index 0-11
            c_sim = self.monthly_param_avg[m, :, 0]
            a_sim = self.monthly_param_avg[m, :, 1]
            eps = self.simulate_residuals(m + 1)[0]  # Resample residuals

            Y_latent[t, :] = c_sim + a_sim * Y_latent[t - 1, :] + eps*0.001
            
        return Y_latent


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == '__main__':
    num_years_sim = 50
    T_sim = num_years_sim * 365
    num_variables = 3  

    true_model = IMAGEModelMatrix(num_variables=num_variables)

    # True monthly parameters
    true_params = np.empty((12, num_variables, 2))
    rng = np.random.default_rng(102)
    for m in range(12):
        for s in range(num_variables):
            true_params[m, s, 0] = 0*(rng.uniform(-0.1, 0.1) + 0.4 * np.cos(m / 12 * 2 * np.pi)) * (s - 1) / 2
            true_params[m, s, 1] = (rng.uniform(0.7, 0.75) + 0.4 * np.cos(m / 12 * 2 * np.pi)) * (1 / (s + 1))

    true_model.monthly_param_avg = true_params.copy()
    true_model.simulate_parameters = lambda: true_model.monthly_param_avg

    for m in range(1, 13):
        true_model.residuals[m] = np.random.normal(0, 0.5, (200, num_variables))

    initial_condition = np.zeros(num_variables)
    Y_sim = true_model.simulate(n_years=num_years_sim, initial_condition=initial_condition)

    start_date_sim = datetime.datetime(2050, 1, 1)
    dates_sim = [start_date_sim + datetime.timedelta(days=i) for i in range(T_sim)]

    fitted_model = IMAGEModelMatrix(num_variables=num_variables)
    fitted_model.fit(Y_sim, dates_sim)

    # --- Step 4: Compare True and Fitted Parameters ---
    true_params_est = true_model.monthly_param_avg
    fitted_params = fitted_model.monthly_param_avg
    months = np.arange(1, 13)

    import matplotlib.cm as cm
    colors = cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for s in range(num_variables):
        axes[0].scatter(months, true_params_est[:, s, 0], label=f"Var {s+1} True", marker='o', color=colors[s])
        axes[0].scatter(months, fitted_params[:, s, 0], label=f"Var {s+1} Fitted", marker='x', color=colors[s])
    
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("AR Intercept (c)")
    axes[0].set_title("Monthly AR Intercept: True vs Fitted")
    axes[0].legend()

    for s in range(num_variables):
        axes[1].scatter(months, true_params_est[:, s, 1], label=f"Var {s+1} True", marker='o', color=colors[s])
        axes[1].scatter(months, fitted_params[:, s, 1], label=f"Var {s+1} Fitted", marker='x', color=colors[s])

    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("AR Coefficient (a)")
    axes[1].set_title("Monthly AR Coefficient: True vs Fitted")
    axes[1].legend()
    
    plt.savefig("figures/true_vs_fitted_params.png")
