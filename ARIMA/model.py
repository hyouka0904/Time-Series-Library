import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from pmdarima import auto_arima
warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA model wrapper for time series forecasting"""
    
    def __init__(self, order=(1,1,1), seasonal_order=(0,0,0,0), auto_select=True):
        """
        Args:
            order: (p,d,q) order of the model
            seasonal_order: (P,D,Q,m) seasonal order
            auto_select: whether to use auto_arima to select best parameters
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_select = auto_select
        self.model = None
        self.fitted_model = None
        self.train_data = None
        self.scaler_mean = None
        self.scaler_std = None
        
    def check_stationarity(self, timeseries):
        """Check if time series is stationary using ADF test"""
        result = adfuller(timeseries, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        return result[1] < 0.05  # True if stationary
    
    def normalize_data(self, data):
        """Normalize the data"""
        self.scaler_mean = np.mean(data)
        self.scaler_std = np.std(data)
        if self.scaler_std == 0:
            self.scaler_std = 1
        return (data - self.scaler_mean) / self.scaler_std
    
    def denormalize_data(self, data):
        """Denormalize the data"""
        if self.scaler_mean is None:
            return data
        return data * self.scaler_std + self.scaler_mean
        
    def fit(self, train_data, normalize=True):
        """
        Fit ARIMA model on training data
        
        Args:
            train_data: 1D numpy array or pandas Series
            normalize: whether to normalize the data
        """
        self.train_data = np.array(train_data).flatten()
        
        if normalize:
            self.train_data = self.normalize_data(self.train_data)
        
        if self.auto_select:
            # Use auto_arima to find best parameters
            print("Searching for best ARIMA parameters...")
            self.model = auto_arima(
                self.train_data,
                start_p=0, start_q=0, max_p=5, max_q=5, 
                seasonal=False, stepwise=True, suppress_warnings=True,
                error_action='ignore', trace=False
            )
            self.order = self.model.order
            print(f"Selected ARIMA order: {self.order}")
            self.fitted_model = self.model
        else:
            # Use specified parameters
            self.model = ARIMA(self.train_data, order=self.order)
            self.fitted_model = self.model.fit()
    
    def predict(self, steps, return_confidence=False, alpha=0.05):
        """
        Make predictions
        
        Args:
            steps: number of steps to forecast
            return_confidence: whether to return confidence intervals
            alpha: significance level for confidence intervals
            
        Returns:
            predictions: array of predictions
            conf_int: confidence intervals (if return_confidence=True)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        if self.auto_select:
            predictions = self.fitted_model.predict(n_periods=steps)
            if return_confidence:
                predictions, conf_int = self.fitted_model.predict(
                    n_periods=steps, return_conf_int=True, alpha=alpha
                )
        else:
            forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
            predictions = forecast_result
            if return_confidence:
                # Get confidence intervals from the forecast
                forecast_result = self.fitted_model.get_forecast(steps=steps)
                conf_int = forecast_result.conf_int(alpha=alpha)
                
        # Denormalize if data was normalized
        if self.scaler_mean is not None:
            predictions = self.denormalize_data(predictions)
            if return_confidence:
                conf_int = self.denormalize_data(conf_int)
                
        if return_confidence:
            return predictions, conf_int
        return predictions
    
    def predict_in_sample(self, start=None, end=None):
        """Get in-sample predictions"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        if self.auto_select:
            predictions = self.fitted_model.predict_in_sample(start=start, end=end)
        else:
            predictions = self.fitted_model.fittedvalues
            
        # Denormalize if needed
        if self.scaler_mean is not None:
            predictions = self.denormalize_data(predictions)
            
        return predictions
    
    def get_residuals(self):
        """Get model residuals"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        if self.auto_select:
            residuals = self.fitted_model.resid()
        else:
            residuals = self.fitted_model.resid
            
        return residuals


class MultivarateARIMAWrapper:
    """Wrapper to handle multivariate time series using multiple ARIMA models"""
    
    def __init__(self, n_features, auto_select=True):
        """
        Args:
            n_features: number of features/variables
            auto_select: whether to use auto_arima
        """
        self.n_features = n_features
        self.models = [ARIMAModel(auto_select=auto_select) for _ in range(n_features)]
        
    def fit(self, train_data, normalize=True):
        """
        Fit ARIMA models for each feature
        
        Args:
            train_data: (n_samples, n_features) array
            normalize: whether to normalize each feature
        """
        train_data = np.array(train_data)
        if len(train_data.shape) == 1:
            train_data = train_data.reshape(-1, 1)
            
        for i in range(self.n_features):
            print(f"Fitting ARIMA model for feature {i+1}/{self.n_features}")
            self.models[i].fit(train_data[:, i], normalize=normalize)
            
    def predict(self, steps):
        """
        Make predictions for all features
        
        Args:
            steps: number of steps to forecast
            
        Returns:
            predictions: (steps, n_features) array
        """
        predictions = []
        for i in range(self.n_features):
            pred = self.models[i].predict(steps)
            predictions.append(pred)
            
        return np.column_stack(predictions)
    
    def get_residuals(self):
        """Get residuals for all features"""
        residuals = []
        for i in range(self.n_features):
            resid = self.models[i].get_residuals()
            residuals.append(resid)
            
        return np.column_stack(residuals)