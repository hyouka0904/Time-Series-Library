import torch
import torch.nn as nn
import numpy as np
import pywt
from statsmodels.tsa.arima.model import ARIMA
from models.TSMixer import Model as TSMixer
import warnings
import pickle
import os
warnings.filterwarnings('ignore')


class Model(nn.Module):
    """
    DWT-ARIMA-TSMixer Hybrid Model (Efficient Version)
    
    This model implements a two-stage approach:
    Stage 1: Fit ARIMA on DWT-decomposed training data (done once)
    Stage 2: Train TSMixer on residuals + approximation components
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.features = configs.features
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # DWT settings
        self.wavelet = getattr(configs, 'wavelet', 'db4')
        self.level = getattr(configs, 'dwt_level', 1)
        
        # ARIMA settings
        self.arima_order = getattr(configs, 'arima_order', (1, 0, 1))
        
        # Initialize TSMixer
        tsmixer_configs = type('configs', (), {})()
        for attr in dir(configs):
            if not attr.startswith('_'):
                setattr(tsmixer_configs, attr, getattr(configs, attr))
        
        self.tsmixer = TSMixer(tsmixer_configs)
        
        # Storage for fitted ARIMA models
        self.arima_models = None
        self.is_fitted = False
        
        # Cache directory for ARIMA models
        self.cache_dir = getattr(configs, 'cache_dir', './arima_cache/')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _dwt_decompose(self, x):
        """Apply DWT decomposition"""
        B, L, D = x.shape
        approx = torch.zeros_like(x)
        detail = torch.zeros_like(x)
        
        x_np = x.detach().cpu().numpy()
        
        for b in range(B):
            for d in range(D):
                coeffs = pywt.wavedec(x_np[b, :, d], self.wavelet, level=self.level)
                
                approx_coeffs = [coeffs[0]] + [None] * self.level
                detail_coeffs = [None] + coeffs[1:]
                
                approx_np = pywt.waverec(approx_coeffs, self.wavelet)
                detail_np = pywt.waverec(detail_coeffs, self.wavelet)
                
                if len(approx_np) > L:
                    approx_np = approx_np[:L]
                    detail_np = detail_np[:L]
                elif len(approx_np) < L:
                    pad_len = L - len(approx_np)
                    approx_np = np.pad(approx_np, (0, pad_len), mode='edge')
                    detail_np = np.pad(detail_np, (0, pad_len), mode='edge')
                
                approx[b, :, d] = torch.tensor(approx_np, dtype=torch.float32)
                detail[b, :, d] = torch.tensor(detail_np, dtype=torch.float32)
        
        return approx.to(x.device), detail.to(x.device)
    
    def fit_arima_stage(self, train_data):
        """
        Stage 1: Fit ARIMA models on training data
        This should be called once before training the neural network
        
        Args:
            train_data: Complete training dataset [N, L, D]
        """
        print("Stage 1: Fitting ARIMA models on training data...")
        
        # Check if models are already cached
        cache_file = os.path.join(self.cache_dir, f'arima_models_{self.arima_order}.pkl')
        if os.path.exists(cache_file):
            print("Loading cached ARIMA models...")
            with open(cache_file, 'rb') as f:
                self.arima_models = pickle.load(f)
            self.is_fitted = True
            return
        
        N, L, D = train_data.shape
        self.arima_models = {}
        
        # Decompose entire training data
        _, detail_all = self._dwt_decompose(train_data)
        detail_np = detail_all.detach().cpu().numpy()
        
        # Fit ARIMA for each feature dimension
        for d in range(D):
            print(f"Fitting ARIMA for feature {d+1}/{D}...")
            
            # Aggregate all samples for this feature
            feature_data = detail_np[:, :, d].flatten()
            
            try:
                model = ARIMA(feature_data, order=self.arima_order)
                fitted_model = model.fit()
                self.arima_models[d] = fitted_model
            except Exception as e:
                print(f"ARIMA failed for feature {d}: {str(e)}")
                self.arima_models[d] = None
        
        # Cache the fitted models
        with open(cache_file, 'wb') as f:
            pickle.dump(self.arima_models, f)
        
        self.is_fitted = True
        print("Stage 1 complete: ARIMA models fitted and cached")
    
    def _get_arima_predictions(self, detail, pred_len):
        """
        Get ARIMA predictions using pre-fitted models
        """
        B, L, D = detail.shape
        arima_pred = torch.zeros(B, pred_len, D)
        residuals = torch.zeros_like(detail)
        
        detail_np = detail.detach().cpu().numpy()
        
        for d in range(D):
            if self.arima_models.get(d) is not None:
                # Use pre-fitted model for forecasting
                fitted_model = self.arima_models[d]
                
                # For each sample in batch
                for b in range(B):
                    try:
                        # Get in-sample predictions
                        # Note: This is a simplified approach
                        # In practice, you might need to update the model with new data
                        in_sample_pred = fitted_model.fittedvalues[-L:]
                        if len(in_sample_pred) < L:
                            in_sample_pred = np.pad(in_sample_pred, (L-len(in_sample_pred), 0), mode='mean')
                        
                        residuals[b, :, d] = torch.tensor(
                            detail_np[b, :, d] - in_sample_pred, 
                            dtype=torch.float32
                        )
                        
                        # Forecast
                        forecast = fitted_model.forecast(steps=pred_len)
                        arima_pred[b, :, d] = torch.tensor(forecast, dtype=torch.float32)
                        
                    except:
                        # Fallback to mean
                        mean_val = detail_np[b, :, d].mean()
                        residuals[b, :, d] = torch.tensor(
                            detail_np[b, :, d] - mean_val, 
                            dtype=torch.float32
                        )
                        arima_pred[b, :, d] = mean_val
            else:
                # Use mean prediction if ARIMA failed
                for b in range(B):
                    mean_val = detail_np[b, :, d].mean()
                    residuals[b, :, d] = torch.tensor(
                        detail_np[b, :, d] - mean_val, 
                        dtype=torch.float32
                    )
                    arima_pred[b, :, d] = mean_val
        
        return arima_pred.to(detail.device), residuals.to(detail.device)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Stage 2: Use pre-fitted ARIMA + TSMixer for forecasting
        """
        if not self.is_fitted:
            raise RuntimeError("ARIMA models not fitted. Call fit_arima_stage() first!")
        
        B, L, D = x_enc.shape
        
        # DWT decomposition
        approx, detail = self._dwt_decompose(x_enc)
        
        # Get ARIMA predictions using pre-fitted models
        arima_pred, residuals = self._get_arima_predictions(detail, self.pred_len)
        
        # TSMixer input: residuals + approximation
        tsmixer_input = residuals + approx
        
        # Apply TSMixer
        tsmixer_output = self.tsmixer.forecast(
            tsmixer_input, x_mark_enc, x_dec, x_mark_dec, mask
        )
        
        # Combine predictions
        final_pred = arima_pred + tsmixer_output
        
        if self.features == 'MS':
            final_pred = final_pred[:, :, -1:]
        
        return final_pred
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise ValueError('Only forecast tasks implemented yet')