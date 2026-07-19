import numpy as np
from scipy import stats
import pandas as pd

def display_regression_results(model, X, y, feature_names=None):
    """
    Display comprehensive results for sklearn LinearRegression model.
    
    Parameters:
    -----------
    model : sklearn.linear_model.LinearRegression
        Fitted linear regression model
    X : array-like or DataFrame of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    feature_names : list, optional
        Names of features (defaults to X.columns if X is DataFrame, else X0, X1, etc.)
    """
    # Convert X to numpy array if it's a DataFrame and store column names
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X_array = X.values
    else:
        X_array = np.array(X)
        if feature_names is None:
            feature_names = [f'X{i}' for i in range(X_array.shape[1])]
            # Update model feature names to avoid warning
            model.feature_names_in_ = np.array(feature_names)
    
    # Add column of ones for intercept
    X_with_intercept = np.column_stack([np.ones(len(X_array)), X_array])
    
    # Get predictions and residuals
    y_pred = model.predict(X_array)
    residuals = y - y_pred
    
    # Calculate basic statistics
    n = X_array.shape[0]  # number of observations
    p = X_array.shape[1]  # number of predictors
    dof = n - p - 1  # degrees of freedom
    
    # Calculate R-squared and adjusted R-squared
    r2 = model.score(X_array, y)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Calculate MSE and RMSE
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate standard errors of coefficients
    mse = np.sum(residuals ** 2) / dof
    covar_matrix = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept))
    std_errors = np.sqrt(np.diagonal(covar_matrix * mse))
    
    # Calculate t-statistics and p-values for all coefficients (including intercept)
    all_coef = np.concatenate(([model.intercept_], model.coef_))
    t_stats = all_coef / std_errors
    p_values = 2 * (1 - stats.t.cdf(abs(t_stats), dof))
    
    # F-statistic
    f_stat = (r2 / p) / ((1 - r2) / dof)
    f_p_value = 1 - stats.f.cdf(f_stat, p, dof)
    
    def get_significance_markers(p_value):
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        elif p_value < 0.1:
            return "."
        return " "
    
    # Print results
    print('='*70)
    print('Linear Regression Results')
    print('='*70)
    print(f'Dependent Variable: y')
    print(f'Observations: {n}')
    print(f'R-squared: {r2:.4f}')
    print(f'Adjusted R-squared: {adj_r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'F-statistic: {f_stat:.4f} (p-value: {f_p_value:.4e})')
    print('\nCoefficients:')
    print('-'*70)
    print(f'{"Variable":<15} {"Coefficient":>12} {"Std Error":>12} {"t-stat":>12} {"P-value":>12} {"":>4}')
    print('-'*70)
    
    # Print intercept
    sig_marker = get_significance_markers(p_values[0])
    print(f'{"Intercept":<15} {model.intercept_:12.4f} {std_errors[0]:12.4f} '
          f'{t_stats[0]:12.4f} {p_values[0]:12.4e} {sig_marker:>4}')
    
    # Print coefficients for each feature
    for name, coef, std_err, t_stat, p_val in zip(
        feature_names, model.coef_, std_errors[1:], t_stats[1:], p_values[1:]):
        sig_marker = get_significance_markers(p_val)
        print(f'{name:<15} {coef:12.4f} {std_err:12.4f} {t_stat:12.4f} {p_val:12.4e} {sig_marker:>4}')
    
    print('\nSignificance codes: 0 "***" 0.001 "**" 0.01 "*" 0.05 "." 0.1 " " 1')
    
    return {
        'r2': r2,
        'adj_r2': adj_r2,
        'rmse': rmse,
        'f_stat': f_stat,
        'f_p_value': f_p_value,
        'coefficients': dict(zip(['intercept'] + feature_names, all_coef)),
        'std_errors': dict(zip(['intercept'] + feature_names, std_errors)),
        't_stats': dict(zip(['intercept'] + feature_names, t_stats)),
        'p_values': dict(zip(['intercept'] + feature_names, p_values))
    }