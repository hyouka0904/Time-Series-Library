import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(pred, true):
    """Calculate various metrics for evaluation"""
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    
    # MAPE
    mask = true != 0
    mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    
    # MSPE
    mspe = np.mean(np.square((true[mask] - pred[mask]) / true[mask])) * 100
    
    return mae, mse, rmse, mape, mspe


def adjustment(gt, pred):
    """Adjustment for anomaly detection (from original code)"""
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def visual(true, pred, name):
    """
    Visualization function for time series
    
    Args:
        true: ground truth
        pred: prediction 
        name: file name to save
    """
    plt.figure(figsize=(12, 4))
    plt.plot(true, label='Ground Truth', color='blue', alpha=0.7)
    plt.plot(pred, label='Prediction', color='red', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.close()


def create_sliding_windows(data, input_len, output_len, stride=1):
    """
    Create sliding windows for time series data
    
    Args:
        data: input data
        input_len: length of input window
        output_len: length of output window
        stride: stride for sliding window
        
    Returns:
        X: input windows
        y: output windows
    """
    X, y = [], []
    for i in range(0, len(data) - input_len - output_len + 1, stride):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len:i + input_len + output_len])
    return np.array(X), np.array(y)


def select_best_arima_order(data, max_p=5, max_d=2, max_q=5):
    """
    Grid search to find best ARIMA parameters using AIC
    
    Args:
        data: time series data
        max_p, max_d, max_q: maximum values for grid search
        
    Returns:
        best_order: tuple of (p, d, q)
        best_aic: best AIC value
    """
    from statsmodels.tsa.arima.model import ARIMA
    import itertools
    import warnings
    warnings.filterwarnings('ignore')
    
    best_aic = np.inf
    best_order = (0, 0, 0)
    
    # Grid search
    for p, d, q in itertools.product(range(max_p + 1), 
                                    range(max_d + 1), 
                                    range(max_q + 1)):
        if p == 0 and q == 0:
            continue
            
        try:
            model = ARIMA(data, order=(p, d, q))
            fitted_model = model.fit()
            aic = fitted_model.aic
            
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                
        except:
            continue
    
    return best_order, best_aic


def check_stationarity(data, significance_level=0.05):
    """
    Check stationarity using Augmented Dickey-Fuller test
    
    Args:
        data: time series data
        significance_level: significance level for test
        
    Returns:
        is_stationary: boolean indicating if series is stationary
        adf_result: detailed test results
    """
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(data, autolag='AIC')
    
    adf_result = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'is_stationary': result[1] < significance_level
    }
    
    return adf_result['is_stationary'], adf_result


def difference_series(data, d=1):
    """
    Difference time series data
    
    Args:
        data: time series data
        d: differencing order
        
    Returns:
        differenced data
    """
    diff = data.copy()
    for _ in range(d):
        diff = np.diff(diff)
    return diff


def inverse_difference(forecast, history, d=1):
    """
    Inverse difference transformation
    
    Args:
        forecast: forecasted values
        history: historical values used for differencing
        d: differencing order
        
    Returns:
        inverted forecast
    """
    # This is a simplified version
    # For proper implementation, would need to track all differencing levels
    if d == 1:
        return np.cumsum(np.concatenate([[history[-1]], forecast]))[1:]
    else:
        # For higher order differencing, need more complex logic
        result = forecast
        for _ in range(d):
            result = np.cumsum(np.concatenate([[history[-1]], result]))[1:]
            history = np.concatenate([history, [result[0]]])
        return result