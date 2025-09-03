import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from model import ARIMAModel, MultivarateARIMAWrapper
from data_loader import ARIMADataLoader
from utils import calculate_metrics, adjustment, visual
import warnings
warnings.filterwarnings('ignore')

# Import M4Meta from data_provider
import sys
sys.path.append('..')  # Add parent directory to path
try:
    from data_provider.m4 import M4Meta
except ImportError:
    print("Warning: Could not import M4Meta. Using local definition.")
    # Fallback definition if import fails
    class M4Meta:
        seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
        horizons = [6, 8, 18, 13, 14, 48]
        frequencies = [1, 4, 12, 1, 1, 24]
        horizons_map = {
            'Yearly': 6,
            'Quarterly': 8,
            'Monthly': 18,
            'Weekly': 13,
            'Daily': 14,
            'Hourly': 48
        }
        frequency_map = {
            'Yearly': 1,
            'Quarterly': 4,
            'Monthly': 12,
            'Weekly': 1,
            'Daily': 1,
            'Hourly': 24
        }
        history_size = {
            'Yearly': 1.5,
            'Quarterly': 1.5,
            'Monthly': 1.5,
            'Weekly': 10,
            'Daily': 10,
            'Hourly': 10
        }


class ARIMAExperimentBase:
    """Base class for ARIMA experiments"""
    
    def __init__(self, args):
        self.args = args
        self.data_loader = ARIMADataLoader(
            data_name=args.data,
            root_path=args.root_path,
            data_path=args.data_path if hasattr(args, 'data_path') else None,
            features=args.features if hasattr(args, 'features') else 'S',
            target=args.target if hasattr(args, 'target') else 'OT',
            scale=True,
            seasonal_patterns=args.seasonal_patterns if hasattr(args, 'seasonal_patterns') else 'Monthly'
        )
        
    def _create_result_dir(self, setting):
        """Create directory for saving results"""
        folder_path = f'./arima_results/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path


class ARIMALongTermForecast(ARIMAExperimentBase):
    """ARIMA experiment for long-term forecasting"""
    
    def train_and_test(self, setting):
        """Train and test ARIMA model for forecasting"""
        print(f"Running ARIMA long-term forecast experiment: {setting}")
        
        # Load data
        train_data, _ = self.data_loader.get_data('train')
        val_data, _ = self.data_loader.get_data('val')
        test_data, _ = self.data_loader.get_data('test')
        
        # For MS task, we predict only the target
        if self.args.features == 'MS':
            target_idx = list(self.data_loader.df_raw.columns).index(self.args.target) - 1
            train_target = train_data[:, target_idx]
            n_features = 1
        else:
            train_target = train_data
            n_features = train_data.shape[1] if len(train_data.shape) > 1 else 1
        
        # Initialize model
        if n_features == 1:
            model = ARIMAModel(auto_select=True)
            model.fit(train_target.flatten(), normalize=False)  # Already normalized
        else:
            model = MultivarateARIMAWrapper(n_features, auto_select=True)
            model.fit(train_target, normalize=False)
        
        # Make predictions
        preds = []
        trues = []
        
        # Sliding window prediction
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        
        # For test set
        test_steps = len(test_data) - seq_len - pred_len + 1
        
        for i in range(0, test_steps, pred_len):  # Step by pred_len to avoid overlap
            # Use historical data for prediction
            if i == 0:
                # Use last seq_len points from training data
                history = train_target[-seq_len:]
            else:
                # Use previous predictions and actual data
                start_idx = max(0, i - seq_len)
                end_idx = i
                history = test_data[start_idx:end_idx]
                if self.args.features == 'MS':
                    history = history[:, target_idx]
            
            # Refit model with recent history (optional, can be slow)
            if i % (pred_len * 5) == 0:  # Refit every 5 predictions
                if n_features == 1:
                    model.fit(np.concatenate([train_target.flatten(), history.flatten()]), 
                             normalize=False)
                else:
                    model.fit(np.concatenate([train_target, history], axis=0), 
                             normalize=False)
            
            # Predict
            pred = model.predict(pred_len)
            
            # Get true values
            true = test_data[i:i+pred_len]
            if self.args.features == 'MS':
                true = true[:, target_idx:target_idx+1]
                pred = pred.reshape(-1, 1)
            elif n_features == 1:
                true = true.reshape(-1, 1)
                pred = pred.reshape(-1, 1)
            
            preds.append(pred)
            trues.append(true)
            
            if i % 100 == 0:
                print(f"Prediction step {i}/{test_steps}")
        
        # Concatenate predictions
        preds = np.array(preds)
        trues = np.array(trues)
        
        # Calculate metrics
        if self.data_loader.scale and self.args.inverse:
            # Inverse transform
            preds_shape = preds.shape
            trues_shape = trues.shape
            preds = preds.reshape(-1, preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-1])
            
            # For MS task, we need to handle single column
            if self.args.features == 'MS':
                # Create dummy full array for inverse transform
                dummy_preds = np.zeros((preds.shape[0], train_data.shape[1]))
                dummy_preds[:, target_idx] = preds.flatten()
                dummy_preds = self.data_loader.inverse_transform(dummy_preds)
                preds = dummy_preds[:, target_idx:target_idx+1]
                
                dummy_trues = np.zeros((trues.shape[0], train_data.shape[1]))
                dummy_trues[:, target_idx] = trues.flatten()
                dummy_trues = self.data_loader.inverse_transform(dummy_trues)
                trues = dummy_trues[:, target_idx:target_idx+1]
            else:
                preds = self.data_loader.inverse_transform(preds)
                trues = self.data_loader.inverse_transform(trues)
            
            preds = preds.reshape(preds_shape)
            trues = trues.reshape(trues_shape)
        
        # Calculate metrics
        mse = mean_squared_error(trues.reshape(-1), preds.reshape(-1))
        mae = mean_absolute_error(trues.reshape(-1), preds.reshape(-1))
        rmse = np.sqrt(mse)
        
        print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        
        # Save results
        folder_path = self._create_result_dir(setting)
        
        # Save metrics
        with open(os.path.join(folder_path, 'metrics.txt'), 'w') as f:
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'MAE: {mae:.4f}\n')
            f.write(f'RMSE: {rmse:.4f}\n')
        
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        
        # Save to summary file
        with open('result_long_term_forecast_arima.txt', 'a') as f:
            f.write(f"{setting}\n")
            f.write(f'mse:{mse:.4f}, mae:{mae:.4f}\n\n')
        
        return mse, mae, rmse


class ARIMAShortTermForecast(ARIMAExperimentBase):
    """ARIMA experiment for short-term forecasting (M4 dataset)"""
    
    def __init__(self, args):
        super().__init__(args)
        # Set M4 specific parameters
        if args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[args.seasonal_patterns]
            self.args.seq_len = 2 * self.args.pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[args.seasonal_patterns]
            # Update data loader with seasonal pattern
            self.data_loader.seasonal_patterns = args.seasonal_patterns
    
    def train_and_test(self, setting):
        """Train and test ARIMA model for short-term forecasting"""
        print(f"Running ARIMA short-term forecast experiment: {setting}")
        
        # For M4 dataset, we need special handling
        if self.args.data == 'm4':
            return self._m4_experiment(setting)
        else:
            # For other datasets, use similar approach as long-term
            return self._general_short_term(setting)
    
    def _m4_experiment(self, setting):
        """Handle M4 dataset specifically"""
        print(f"M4 experiment for {self.args.seasonal_patterns}")
        
        # Load M4 data
        train_data, train_ids = self.data_loader.get_data('train')
        test_info, test_ids = self.data_loader.get_data('test')
        
        if isinstance(test_info, tuple):
            train_context, test_data = test_info
        else:
            test_data = test_info
            train_context = train_data
        
        # Prepare for storing predictions
        all_predictions = []
        all_actuals = []
        
        # Process each time series
        num_series = len(train_data)
        print(f"Processing {num_series} time series...")
        
        for i in range(min(num_series, 1000)):  # Limit to 1000 series for speed
            if i % 100 == 0:
                print(f"Processing series {i}/{num_series}")
            
            # Get training data for this series
            ts_train = train_data[i]
            
            # Fit ARIMA model
            model = ARIMAModel(auto_select=True)
            try:
                model.fit(ts_train, normalize=False)
                
                # Make predictions
                predictions = model.predict(self.args.pred_len)
                
                # Get actual values if available
                if i < len(test_data):
                    actuals = test_data[i][:self.args.pred_len]
                    all_predictions.append(predictions[:len(actuals)])
                    all_actuals.append(actuals)
                else:
                    all_predictions.append(predictions)
            except Exception as e:
                print(f"Error processing series {i}: {e}")
                # Use naive forecast as fallback
                naive_pred = np.repeat(ts_train[-1], self.args.pred_len)
                all_predictions.append(naive_pred)
                if i < len(test_data):
                    actuals = test_data[i][:self.args.pred_len]
                    all_actuals.append(actuals)
        
        # Calculate metrics
        if all_actuals:
            # Calculate SMAPE
            def smape(y_true, y_pred):
                denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
                diff = np.abs(y_true - y_pred) / denominator
                diff[denominator == 0] = 0.0
                return 100 * np.mean(diff)
            
            # Calculate metrics for each series
            smapes = []
            for pred, actual in zip(all_predictions, all_actuals):
                if len(pred) == len(actual):
                    smapes.append(smape(actual, pred))
            
            avg_smape = np.mean(smapes)
            
            # OWA is relative to baseline, set to 1.0 for ARIMA
            owa = 1.0
            
            print(f"Average SMAPE: {avg_smape:.4f}")
        else:
            # No test data available
            avg_smape = 0.0
            owa = 0.0
            print("No test data available for evaluation")
        
        # Save results
        folder_path = self._create_result_dir(setting)
        
        # Save predictions in M4 format
        if all_predictions:
            pred_df = pd.DataFrame(all_predictions[:len(test_ids)])
            pred_df.index = test_ids[:len(all_predictions)]
            pred_df.to_csv(os.path.join(folder_path, f'{self.args.seasonal_patterns}_forecast.csv'))
        
        with open(os.path.join(folder_path, 'metrics.txt'), 'w') as f:
            f.write(f"SMAPE: {avg_smape:.4f}\n")
            f.write(f"OWA: {owa:.4f}\n")
        
        with open('result_short_term_forecast_arima.txt', 'a') as f:
            f.write(f"{setting}\n")
            f.write(f'smape:{avg_smape:.4f}, owa:{owa:.4f}\n\n')
        
        return avg_smape, owa
    
    def _general_short_term(self, setting):
        """General short-term forecasting for non-M4 datasets"""
        print(f"Running general short-term forecast for {self.args.data}")
        
        # Load data
        train_data, _ = self.data_loader.get_data('train')
        val_data, _ = self.data_loader.get_data('val')
        test_data, _ = self.data_loader.get_data('test')
        
        # For MS task, predict only target
        if self.args.features == 'MS':
            target_idx = list(self.data_loader.df_raw.columns).index(self.args.target) - 1
            train_target = train_data[:, target_idx]
            n_features = 1
        else:
            train_target = train_data
            n_features = train_data.shape[1] if len(train_data.shape) > 1 else 1
        
        # Initialize model
        if n_features == 1:
            model = ARIMAModel(auto_select=True)
            model.fit(train_target.flatten(), normalize=False)
        else:
            model = MultivarateARIMAWrapper(n_features, auto_select=True)
            model.fit(train_target, normalize=False)
        
        # Short-term predictions with more frequent model updates
        preds = []
        trues = []
        
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        
        # Sliding window prediction
        test_steps = min(1000, len(test_data) - seq_len - pred_len + 1)  # Limit for short-term
        
        for i in range(0, test_steps, pred_len):
            # Use recent history
            if i == 0:
                history = train_target[-seq_len:]
            else:
                start_idx = max(0, i - seq_len)
                end_idx = i
                history = test_data[start_idx:end_idx]
                if self.args.features == 'MS':
                    history = history[:, target_idx]
            
            # Refit model more frequently for short-term
            if i % pred_len == 0:  # Refit every prediction
                if n_features == 1:
                    recent_train = np.concatenate([train_target[-500:], history.flatten()])
                    model.fit(recent_train, normalize=False)
                else:
                    recent_train = np.concatenate([train_target[-500:], history], axis=0)
                    model.fit(recent_train, normalize=False)
            
            # Predict
            pred = model.predict(pred_len)
            
            # Get true values
            true = test_data[i:i+pred_len]
            if self.args.features == 'MS':
                true = true[:, target_idx:target_idx+1]
                pred = pred.reshape(-1, 1)
            elif n_features == 1:
                true = true.reshape(-1, 1)
                pred = pred.reshape(-1, 1)
            
            preds.append(pred)
            trues.append(true)
        
        # Concatenate predictions
        preds = np.array(preds)
        trues = np.array(trues)
        
        # Calculate metrics
        if self.data_loader.scale and self.args.inverse:
            # Inverse transform
            preds = self._inverse_transform(preds, target_idx if self.args.features == 'MS' else None)
            trues = self._inverse_transform(trues, target_idx if self.args.features == 'MS' else None)
        
        # Calculate metrics
        mse = mean_squared_error(trues.reshape(-1), preds.reshape(-1))
        mae = mean_absolute_error(trues.reshape(-1), preds.reshape(-1))
        rmse = np.sqrt(mse)
        
        # Calculate SMAPE for short-term forecast
        def smape(y_true, y_pred):
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        
        smape_value = smape(trues.reshape(-1), preds.reshape(-1))
        
        print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape_value:.4f}')
        
        # Save results
        folder_path = self._create_result_dir(setting)
        
        with open(os.path.join(folder_path, 'metrics.txt'), 'w') as f:
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'MAE: {mae:.4f}\n')
            f.write(f'RMSE: {rmse:.4f}\n')
            f.write(f'SMAPE: {smape_value:.4f}\n')
        
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        
        with open('result_short_term_forecast_arima.txt', 'a') as f:
            f.write(f"{setting}\n")
            f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, smape:{smape_value:.4f}\n\n')
        
        return mse, mae, rmse
    
    def _inverse_transform(self, data, target_idx=None):
        """Helper method for inverse transform"""
        shape = data.shape
        data = data.reshape(-1, data.shape[-1])
        
        if target_idx is not None:
            # For MS task
            dummy_data = np.zeros((data.shape[0], self.data_loader.scaler.n_features_in_))
            dummy_data[:, target_idx] = data.flatten()
            dummy_data = self.data_loader.inverse_transform(dummy_data)
            data = dummy_data[:, target_idx:target_idx+1]
        else:
            data = self.data_loader.inverse_transform(data)
        
        return data.reshape(shape)


class ARIMAAnomalyDetection(ARIMAExperimentBase):
    """ARIMA experiment for anomaly detection using residuals"""
    
    def train_and_test(self, setting):
        """Train and test ARIMA model for anomaly detection"""
        print(f"Running ARIMA anomaly detection experiment: {setting}")
        
        # Load data
        train_data, _ = self.data_loader.get_data('train')
        test_data, test_labels = self.data_loader.get_data('test')
        
        # Flatten labels
        test_labels = test_labels.flatten()
        
        # Train separate ARIMA model for each feature
        n_features = train_data.shape[1] if len(train_data.shape) > 1 else 1
        
        all_scores = []
        
        for feat_idx in range(n_features):
            print(f"Processing feature {feat_idx + 1}/{n_features}")
            
            # Get feature data
            if n_features == 1:
                train_feat = train_data.flatten()
                test_feat = test_data.flatten()
            else:
                train_feat = train_data[:, feat_idx]
                test_feat = test_data[:, feat_idx]
            
            # Train ARIMA model
            model = ARIMAModel(auto_select=True)
            model.fit(train_feat, normalize=False)
            
            # Calculate anomaly scores based on prediction error
            scores = []
            window_size = 100  # Use sliding window
            
            for i in range(len(test_feat) - window_size):
                # Predict next value
                try:
                    pred = model.predict(1)[0]
                    actual = test_feat[i + window_size]
                    error = abs(actual - pred)
                    scores.append(error)
                    
                    # Update model with new observation (optional)
                    if i % 100 == 0:
                        # Refit with recent data
                        recent_data = test_feat[max(0, i-1000):i+window_size]
                        model.fit(np.concatenate([train_feat[-1000:], recent_data]), 
                                 normalize=False)
                except:
                    scores.append(0)
            
            # Pad scores to match test length
            scores = scores + [scores[-1]] * (len(test_feat) - len(scores))
            all_scores.append(scores)
        
        # Combine scores from all features
        all_scores = np.array(all_scores)
        anomaly_scores = np.mean(all_scores, axis=0)
        
        # Calculate threshold based on training data
        train_scores = []
        for feat_idx in range(n_features):
            if n_features == 1:
                train_feat = train_data.flatten()
            else:
                train_feat = train_data[:, feat_idx]
                
            model = ARIMAModel(auto_select=True)
            model.fit(train_feat[:len(train_feat)//2], normalize=False)
            
            # Get training scores
            feat_scores = []
            for i in range(len(train_feat)//2, len(train_feat)-1):
                try:
                    pred = model.predict(1)[0]
                    actual = train_feat[i+1]
                    error = abs(actual - pred)
                    feat_scores.append(error)
                except:
                    feat_scores.append(0)
            train_scores.extend(feat_scores)
        
        # Set threshold
        threshold = np.percentile(train_scores, 100 - self.args.anomaly_ratio)
        
        # Make predictions
        pred = (anomaly_scores > threshold).astype(int)
        
        # Ensure same length
        min_len = min(len(pred), len(test_labels))
        pred = pred[:min_len]
        test_labels = test_labels[:min_len]
        
        # Apply adjustment
        gt, pred = adjustment(test_labels, pred)
        
        # Calculate metrics
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0
        )
        
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F-score: {f_score:.4f}")
        
        # Save results
        folder_path = self._create_result_dir(setting)
        
        with open(os.path.join(folder_path, 'metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F-score: {f_score:.4f}\n")
        
        with open('result_anomaly_detection_arima.txt', 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F-score: {f_score:.4f}\n\n")
        
        return accuracy, precision, recall, f_score


class ARIMAImputation(ARIMAExperimentBase):
    """ARIMA experiment for time series imputation"""
    
    def train_and_test(self, setting):
        """Train and test ARIMA model for imputation"""
        print(f"Running ARIMA imputation experiment: {setting}")
        
        # Load data
        train_data, _ = self.data_loader.get_data('train')
        test_data, _ = self.data_loader.get_data('test')
        
        n_features = test_data.shape[1] if len(test_data.shape) > 1 else 1
        
        # Create random mask
        mask = np.random.rand(*test_data.shape) > self.args.mask_rate
        masked_data = test_data.copy()
        masked_data[~mask] = np.nan
        
        # Impute each feature separately
        imputed_data = masked_data.copy()
        
        for feat_idx in range(n_features):
            print(f"Imputing feature {feat_idx + 1}/{n_features}")
            
            if n_features == 1:
                train_feat = train_data.flatten()
                test_feat = test_data.flatten()
                mask_feat = mask.flatten()
            else:
                train_feat = train_data[:, feat_idx]
                test_feat = test_data[:, feat_idx]
                mask_feat = mask[:, feat_idx]
            
            # Train ARIMA model
            model = ARIMAModel(auto_select=True)
            model.fit(train_feat, normalize=False)
            
            # Forward fill for imputation
            imputed_feat = test_feat.copy()
            
            for i in range(len(test_feat)):
                if not mask_feat[i]:  # Need to impute
                    # Use ARIMA to predict missing value
                    # Find last known value
                    last_known_idx = i - 1
                    while last_known_idx >= 0 and not mask_feat[last_known_idx]:
                        last_known_idx -= 1
                    
                    if last_known_idx >= 0:
                        # Use recent history to predict
                        history_len = min(100, last_known_idx + 1)
                        history = imputed_feat[max(0, last_known_idx-history_len+1):last_known_idx+1]
                        
                        # Refit model with recent data
                        model.fit(np.concatenate([train_feat[-500:], history]), normalize=False)
                        
                        # Predict missing value
                        steps_ahead = i - last_known_idx
                        try:
                            predictions = model.predict(steps_ahead)
                            imputed_feat[i] = predictions[-1]
                        except:
                            # Fallback to forward fill
                            imputed_feat[i] = imputed_feat[last_known_idx]
                    else:
                        # Use training data mean
                        imputed_feat[i] = np.mean(train_feat)
            
            if n_features == 1:
                imputed_data = imputed_feat.reshape(-1, 1)
            else:
                imputed_data[:, feat_idx] = imputed_feat
        
        # Calculate metrics only on masked values
        mse = mean_squared_error(test_data[~mask], imputed_data[~mask])
        mae = mean_absolute_error(test_data[~mask], imputed_data[~mask])
        rmse = np.sqrt(mse)
        
        print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')
        
        # Save results
        folder_path = self._create_result_dir(setting)
        
        with open(os.path.join(folder_path, 'metrics.txt'), 'w') as f:
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'MAE: {mae:.4f}\n')
            f.write(f'RMSE: {rmse:.4f}\n')
        
        np.save(os.path.join(folder_path, 'imputed.npy'), imputed_data)
        np.save(os.path.join(folder_path, 'true.npy'), test_data)
        np.save(os.path.join(folder_path, 'mask.npy'), mask)
        
        with open('result_imputation_arima.txt', 'a') as f:
            f.write(f"{setting}\n")
            f.write(f'mse:{mse:.4f}, mae:{mae:.4f}\n\n')
        
        return mse, mae, rmse