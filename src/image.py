import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import datetime


import scipy.optimize as opt

def neg_log_likelihood(params, y):
    """
    Compute the negative log-likelihood for an AR(1) model.
    
    Model: y[t] = c + a * y[t-1] + ε[t],  ε[t] ~ N(0, σ²)
    
    params: [c, a, log_sigma] (we optimize over log_sigma to ensure positivity)
    y: 1D array of latent data.
    """
    c, a, log_sigma = params
    sigma = np.exp(log_sigma)
    residuals = y[1:] - (c + a * y[:-1])
    n = len(residuals)
    nll = n * np.log(sigma) + 0.5 * n * np.log(2 * np.pi) + np.sum(residuals**2) / (2 * sigma**2)
    return nll

def mle_ar1(y):
    """
    Compute MLE estimates for an AR(1) model on 1D latent data y.
    Returns estimated (c, a).
    """
    # Initial guess: use OLS estimates as starting point.
    c0 = np.mean(y[1:] - y[:-1])
    a0 = np.corrcoef(y[:-1], y[1:])[0, 1]
    sigma0 = np.std(y[1:] - (c0 + a0*y[:-1]))
    init = np.array([c0, a0, np.log(sigma0 + 1e-6)])
    # Bounds: a between -1 and 1, no bounds on c, log_sigma free.
    bounds = [(None, None), (-1, 1), (None, None)]
    res = opt.minimize(lambda params: neg_log_likelihood(params, y), init, bounds=bounds)
    if res.success:
        c_hat, a_hat, _ = res.x
        return c_hat, a_hat
    else:
        return np.nan, np.nan

class IMAGEModelMatrix:
    def __init__(self, num_variables):
        """
        Initialize the IMAGE model.
        
        Parameters:
            num_variables: Total number of variables (e.g. Tmax, Tmin, Precip, etc.)
        """
        self.num_variables = num_variables  # S: total number of series
        self.months = 12  # M = 12
        self.P = None     # Number of years in historical data
        # H will be a matrix of shape (P, 2*S*M) where each row is the vectorized monthly parameters
        self.H = None
        # The historical average monthly parameters, shaped (12, S, 2) where for each month we have (c, a) per variable.
        self.monthly_param_avg = None
        # Residuals for each month (aggregated over years); key: month (1...12), value: list of (s, residual)
        self.residuals = {}
        # For inverse NQT: store sorted historical observations for each variable.
        self.empirical = {}

    def normal_quantile_transform(self, X):
        """
        Transform a 1D array X (observed data) to latent Gaussian space using NQT.
        """
        ranks = stats.rankdata(X, method='average') / (X.size + 1)
        return stats.norm.ppf(ranks)

    def inverse_normal_quantile_transform(self, y, sorted_hist):
        """
        Inverse NQT: Given latent value y, use linear interpolation on the empirical CDF.
        """
        p = stats.norm.cdf(y)
        return np.interp(p, np.linspace(0, 1, len(sorted_hist)), sorted_hist)

    def fit(self, X, dates):
        """
        Directly estimate monthly AR(1) parameters by pooling data across all years.
        
        X: array of shape (T, num_variables) (observed data)
        dates: list of datetime objects (length T)
        
        For each month m, pool all data from that month (across years) and estimate:
             y[t] = c(m) + a(m) * y[t-1] + ε[t]
        on the latent space (after applying NQT).
        
        The estimated parameters are stored in self.monthly_param_avg with shape (12, num_variables, 2)
        where [:, s, 0] = c (intercept) and [:, s, 1] = a (AR coefficient).
        """
        T, S = X.shape
        # Save empirical distributions for inverse NQT.
        for s in range(S):
            self.empirical[s] = np.sort(X[:, s])
        monthly_params = np.zeros((12, S, 2))
        # Initialize residuals for each month.
        for m in range(1, 13):
            self.residuals[m] = []
        # Loop over months 1...12
        for m in range(1, 13):
            # Get indices for all days in month m (pooled over years).
            idx_month = [i for i, d in enumerate(dates) if d.month == m]
            if len(idx_month) < 2:
                monthly_params[m-1, :, :] = np.nan
                continue
            X_month = X[idx_month, :]  # shape (T_m, S)
            for s in range(S):
                y = self.normal_quantile_transform(X_month[:, s])
                # Use the MLE (via numerical optimization) to estimate AR(1) parameters.
                c_hat, a_hat = mle_ar1(y)
                monthly_params[m-1, s, 0] = c_hat
                monthly_params[m-1, s, 1] = a_hat
                # Compute residuals.
                for t in range(1, len(y)):
                    eps = y[t] - (c_hat + a_hat * y[t-1])
                    self.residuals[m].append((s, eps))
        self.monthly_param_avg = monthly_params

    def simulate_parameters(self):
        """
        Simulate new monthly parameters using PXEOF resampling.
        
        Following the paper:
          - Vectorize the annual parameter estimates into H (already done).
          - Perform an EOF decomposition (via SVD): H = U Σ V^T.
          - Generate a new realization by perturbing the principal components.
          - Reconstruct simulated parameters and average over years.
        
        For simplicity, we add Gaussian noise in the EOF space.
        Returns:
            simulated_params: array of shape (12, num_variables, 2)
        """
        U, Sigma, VT = np.linalg.svd(self.H, full_matrices=False)
        # Project H onto EOF space: G = U Σ.
        G = U * Sigma  # shape (P, 2*S*12)
        # Add noise to G.
        G_sim = G + np.random.normal(0, 0.05, G.shape)
        # Reconstruct simulated H.
        H_sim = G_sim @ VT  # shape (P, 2*S*12)
        # For simulation, we take the mean over simulated years.
        h_sim_mean = np.mean(H_sim, axis=0)
        simulated_params = h_sim_mean.reshape(self.months, self.num_variables, 2)
        return simulated_params

    def simulate_residuals(self, month, n_steps):
        """
        For a given month (1-12), simulate residuals by resampling from the historical residuals.
        Returns an array of shape (n_steps, num_variables), where for each variable we sample randomly.
        """
        S = self.num_variables
        # Organize residuals by variable.
        res_by_var = {s: [] for s in range(S)}
        for s, eps in self.residuals[month]:
            res_by_var[s].append(eps)
        eps_sim = np.zeros((n_steps, S))
        for s in range(S):
            if len(res_by_var[s]) > 0:
                arr = np.array(res_by_var[s])
                idx = np.random.randint(0, len(arr), size=n_steps)
                eps_sim[:, s] = arr[idx]
            else:
                eps_sim[:, s] = 0.0
        return eps_sim *0.1

    def simulate(self, n_years, initial_condition, hist_data_for_inverse=None):
        """
        Simulate daily latent series using the simulated monthly parameters and resampled residuals.
        
        n_years: number of years to simulate.
        initial_condition: array of shape (num_variables,) giving the initial latent state.
        hist_data_for_inverse: if provided, used for inverse NQT.
        
        Returns:
            Simulated series in observed space (if inverse transform is applied) or latent space.
            Shape: (n_days, num_variables), where n_days = n_years * 365.
        """
        n_days = n_years * 365
        S = self.num_variables
        Y_latent = np.zeros((n_days, S))
        Y_latent[0, :] = initial_condition
        simulated_params = self.simulate_parameters()  # shape (12, S, 2)
        for t in range(1, n_days):
            m = (t // 30) % 12  # m in 0..11 corresponds to month m+1
            c_sim = simulated_params[m, :, 0]
            a_sim = simulated_params[m, :, 1]
            # Resample residual for this day (for month m+1)
            eps = self.simulate_residuals(m+1, 1)[0]  # shape (S,)
            Y_latent[t, :] = c_sim + a_sim * Y_latent[t-1, :] + eps
        # Apply inverse NQT if historical data provided.
        if hist_data_for_inverse is not None:
            Y_obs = np.zeros_like(Y_latent)
            for s in range(S):
                sorted_hist = np.sort(hist_data_for_inverse[:, s])
                Y_obs[:, s] = self.inverse_normal_quantile_transform(Y_latent[:, s], sorted_hist)
            return Y_obs
        else:
            return Y_latent

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == '__main__':
    num_years_sim = 50             # Number of years to simulate
    T_sim = num_years_sim * 365    # Total simulated days
    num_variables = 3            # e.g., Tmax, Tmin, Precip
    
    # --- Step 1. Create a True Model with Preset (Random) Monthly Parameters ---
    true_model = IMAGEModelMatrix(num_variables=num_variables)
    
    # Create random "true" monthly parameters:
    # For each month (1..12) and each variable, let c ~ Uniform(-0.5, 0.5) and a ~ Uniform(0.7, 0.95)
    true_params = np.empty((12, num_variables, 2))
    rng = np.random.default_rng(102)
    for m in range(12):
        for s in range(num_variables):
            true_params[m, s, 0] = (rng.uniform(-0.1, 0.1)+0.4*np.cos(m/12*2*np.pi))*(s-1)/2    # c
            true_params[m, s, 1] = (rng.uniform(0.7, 0.75)+0.1*np.cos(m/12*2*np.pi))*(1/(s+1))    # a
    # Set the true model's monthly average parameters:
    true_model.monthly_param_avg = true_params.copy()
    
    # To allow simulation, override simulate_parameters so it returns our preset true parameters.
    true_model.simulate_parameters = lambda: true_model.monthly_param_avg
    
    # Also, assign residuals for each month. For simplicity, for each month (m=1..12) for each variable,
    # generate 200 samples from N(0, 0.5) and store as (s, sample).
    for m in range(1, 13):
        res_list = []
        for s in range(num_variables):
            samples = np.random.normal(0, 0.5, 200)
            for sample in samples:
                res_list.append((s, sample))
        true_model.residuals[m] = res_list
    
    # --- Step 2. Simulate Data from the True Model ---
    initial_condition = np.zeros(num_variables)  # starting latent state
    # Here we simulate in latent space (we set hist_data_for_inverse=None)
    Y_sim = true_model.simulate(n_years=num_years_sim, initial_condition=initial_condition, 
                                hist_data_for_inverse=None)
    
    # Generate simulated dates starting from January 1, 2050.
    start_date_sim = datetime.datetime(2050, 1, 1)
    dates_sim = [start_date_sim + datetime.timedelta(days=i) for i in range(T_sim)]
    
    # --- Step 3. Fit a New Model to the Simulated Data ---
    fitted_model = IMAGEModelMatrix(num_variables=num_variables)
    # Fit the new model using the simulated data and dates.
    fitted_model.fit(Y_sim, dates_sim)
    
    # --- Step 4. Compare the True and Fitted Monthly Parameters ---
    # True parameters from our true_model
    true_params_est = true_model.monthly_param_avg  # shape: (12, num_variables, 2)
    # Fitted parameters estimated from simulated data.
    fitted_params = fitted_model.monthly_param_avg   # shape: (12, num_variables, 2)
    months = np.arange(1, 13)
    
    import matplotlib.cm as cm
    colors = cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot for AR intercept (c)
    for s in range(num_variables):
        axes[0].scatter(months, true_params_est[:, s, 0], label=f"Var {s+1} True", marker='o', color=colors[s])
        axes[0].scatter(months, fitted_params[:, s, 0], label=f"Var {s+1} Fitted", marker='x', color=colors[s])
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("AR Intercept (c)")
    axes[0].set_title("Monthly AR Intercept: True vs Fitted")
    axes[0].legend()
    
    # Scatter plot for AR coefficient (a)
    for s in range(num_variables):
        axes[1].scatter(months, true_params_est[:, s, 1], label=f"Var {s+1} True", marker='o', color=colors[s])
        axes[1].scatter(months, fitted_params[:, s, 1], label=f"Var {s+1} Fitted", marker='x', color=colors[s])
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("AR Coefficient (a)")
    axes[1].set_title("Monthly AR Coefficient: True vs Fitted")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("figures/true_vs_fitted_params.png")

    #plot the time series
    fig, axes = plt.subplots(num_variables, 1, figsize=(14, 6), sharex=True)
    for s in range(num_variables):
        axes[s].plot(dates_sim, Y_sim[:, s], label=f"Var {s+1} Simulated", color=colors[s])
        axes[s].set_ylabel(f"Var {s+1}")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig("figures/simulated_timeseries.png")