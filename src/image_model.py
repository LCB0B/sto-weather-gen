import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as opt
from sklearn.decomposition import PCA


def neg_log_likelihood(params, y):
    """
    Compute the negative log-likelihood for an AR(1) model. #add other models later
    
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

class IMAGE:
    def __init__(self, num_variables,num_locations):
        """
        Initialize the IMAGE model.
        
        Parameters:
            num_variables (int): Number of variables in the model.
        """
        self.total_num_variables = num_variables * num_locations
        self.num_variables = num_variables
        self.num_locations = num_locations
    
        
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
    
    def fit(self, x, dates):
        """
        Estimate monthly AR(1) parameters across all years.
        
        x: array of shape (T, num_variables*num_locations) (observed data)
        dates: list of datetime objects (length T)
        
        For each month m, pool all data from that month and estimate:
             y_s[t] = c_s + a_s * y_s[t-1] + ε_s[t]
        on the latent space (after applying NQT).
        
        The estimated parameters are stored in self.C with shape (S, M) and self.A with shape (S, M).
        then in self.c_p with shape (S,M,Y) and self.a_p with shape (S,M,Y)
        where S is the total number of variable, M is the number of months and Y is the number of years.
        """

        # go through dates to get the number of years
        num_years = dates[-1].year - dates[0].year + 1
        self.num_years = num_years
        self.months = 12
        self.c_p = np.zeros((self.total_num_variables, self.months, num_years))
        self.a_p = np.zeros((self.total_num_variables, self.months, num_years))

        # apply NQT to the data along the variable axis
        X = self.normal_quantile_transform([x[:, i] for i in range(self.total_num_variables)])

        #loop over years
        for y in range(num_years):
            self.C = np.zeros((self.total_num_variables, self.months))
            self.A = np.zeros((self.total_num_variables, self.months))
            for m in range(1, 13):
                #get the data for the month m
                mask_month = np.array([d.month == m and d.year == dates[0].year + y for d in dates])
                X_month = X[:, mask_month]
                #loop over variables
                for s in range(self.total_num_variables):
                    #get the data for the variable s
                    x_s = X_month[s]
                    #estimate the AR(1) parameters for the variable s
                    self.C[s, m-1], self.A[s, m-1] = mle_ar1(x_s)

            self.c_p[:, :, y] = self.C
            self.a_p[:, :, y] = self.A

            #theta each row represents the estimates of observationparameters for a given year and each column is the annual
            #  time series of a parameter for a given variable at a given
            #  month and P is the number of years of observation data
            self.Theta = np.zeros((self.total_num_variables, 2, num_years))
            self.Theta[:, 0, y] = self.C.flatten()
            self.Theta[:, 1, y] = self.A.flatten()

        return

        def EOF_decompistion(self):
            """
             An EOF decomposition on H yields matrix Lambda whose columns contain the EOF modes and projecting H onto Lambda gives, G = Theta Lambda 
            """
            # EOF Decomposition (SVD)
            U, Sigma, VT = np.linalg.svd(self.Theta, full_matrices=False)
            self.Lambda = U * Sigma
            
            #phi = Lambda . Gamma + mu, mu contains the column means of Theta, and Gamma_{r,i} r is a random variable following the discrete uniform distribution over the set 1:Y
            #and i is the index of the EOF mode
            self.mu = np.mean(self.Theta, axis = 2)
            self.Gamma = np.random.randint(0, self.Theta.shape[2], self.Theta.shape[1])
            self.Phi = np.linalg(self.Lambda,self.Gamma) + self.mu
            return self.Phi
        
        def simulate_parameters(self, month):
            """
            Resample residuals using bivariate periodically extended empirical orthogonal functions (PXEOF) resampling from the Phi matrix.
            """
            C = self.Phi[: self.total_num_variables]
            A = self.Phi[self.total_num_variables:]

            c_p = C.reshape(self.total_num_variables, self.months, self.num_years) 
            a_p = A.reshape(self.total_num_variables, self.months, self.num_years)\
            
        def simulate_residuals(self, month):
            """
            Resample residuals using EOF-based resampling.

            Inputs:
                - month: The month (1-12) to sample residuals from.

            Outputs:
                - Simulated residual vector for that month.
            """
            S = self.num_variables
            E = np.array(self.residuals[month]).reshape(-1, S)

