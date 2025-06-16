import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ARIMADataLoader:
    """Data loader for ARIMA experiments"""
    
    def __init__(self, data_name, root_path, data_path=None, 
                 features='S', target='OT', scale=True):
        """
        Args:
            data_name: dataset name (ETTh1, ETTm1, custom, etc.)
            root_path: root path of data files
            data_path: specific data file path
            features: M/S/MS (multivariate/univariate)
            target: target column for S/MS tasks
            scale: whether to scale the data
        """
        self.data_name = data_name
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = StandardScaler()
        
    def load_ETT_data(self, flag='train'):
        """Load ETT dataset"""
        if self.data_path is None:
            self.data_path = f'{self.data_name}.csv'
            
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Define train/val/test split based on the dataset type
        if 'ETTh' in self.data_name:
            border1s = [0, 12*30*24, 12*30*24+4*30*24]
            border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        else:  # ETTm
            border1s = [0, 12*30*24*4, 12*30*24*4+4*30*24*4]
            border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
            
        type_map = {'train': 0, 'val': 1, 'test': 2}
        border1 = border1s[type_map[flag]]
        border2 = border2s[type_map[flag]]
        
        # Select features
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            
        # Scale data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # Get the specific split
        data = data[border1:border2]
        
        # Also get timestamp information
        df_stamp = df_raw[['date']][border1:border2]
        
        return data, df_stamp
    
    def load_custom_data(self, flag='train'):
        """Load custom dataset"""
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Reorder columns
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # Define train/val/test split
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        border1 = border1s[type_map[flag]]
        border2 = border2s[type_map[flag]]
        
        # Select features
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            
        # Scale data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # Get the specific split
        data = data[border1:border2]
        df_stamp = df_raw[['date']][border1:border2]
        
        return data, df_stamp
    
    def load_anomaly_data(self, flag='train'):
        """Load anomaly detection datasets"""
        if self.data_name == 'PSM':
            return self._load_PSM(flag)
        elif self.data_name == 'MSL':
            return self._load_MSL(flag)
        elif self.data_name == 'SMAP':
            return self._load_SMAP(flag)
        elif self.data_name == 'SMD':
            return self._load_SMD(flag)
        elif self.data_name == 'SWAT':
            return self._load_SWAT(flag)
        else:
            raise ValueError(f"Unknown anomaly dataset: {self.data_name}")
    
    def _load_PSM(self, flag):
        """Load PSM dataset"""
        data = pd.read_csv(os.path.join(self.root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = pd.read_csv(os.path.join(self.root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_data = self.scaler.transform(test_data)
        
        test_labels = pd.read_csv(os.path.join(self.root_path, 'test_label.csv')).values[:, 1:]
        
        data_len = len(data)
        val_data = data[int(data_len * 0.8):]
        train_data = data[:int(data_len * 0.8)]
        
        if flag == 'train':
            return train_data, None
        elif flag == 'val':
            return val_data, None
        else:
            return test_data, test_labels
            
    def _load_MSL(self, flag):
        """Load MSL dataset"""
        data = np.load(os.path.join(self.root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = np.load(os.path.join(self.root_path, "MSL_test.npy"))
        test_data = self.scaler.transform(test_data)
        test_labels = np.load(os.path.join(self.root_path, "MSL_test_label.npy"))
        
        data_len = len(data)
        val_data = data[int(data_len * 0.8):]
        train_data = data[:int(data_len * 0.8)]
        
        if flag == 'train':
            return train_data, None
        elif flag == 'val':
            return val_data, None
        else:
            return test_data, test_labels
            
    def _load_SMAP(self, flag):
        """Load SMAP dataset"""
        data = np.load(os.path.join(self.root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = np.load(os.path.join(self.root_path, "SMAP_test.npy"))
        test_data = self.scaler.transform(test_data)
        test_labels = np.load(os.path.join(self.root_path, "SMAP_test_label.npy"))
        
        data_len = len(data)
        val_data = data[int(data_len * 0.8):]
        train_data = data[:int(data_len * 0.8)]
        
        if flag == 'train':
            return train_data, None
        elif flag == 'val':
            return val_data, None
        else:
            return test_data, test_labels
            
    def _load_SMD(self, flag):
        """Load SMD dataset"""
        data = np.load(os.path.join(self.root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = np.load(os.path.join(self.root_path, "SMD_test.npy"))
        test_data = self.scaler.transform(test_data)
        test_labels = np.load(os.path.join(self.root_path, "SMD_test_label.npy"))
        
        data_len = len(data)
        val_data = data[int(data_len * 0.8):]
        train_data = data[:int(data_len * 0.8)]
        
        if flag == 'train':
            return train_data, None
        elif flag == 'val':
            return val_data, None
        else:
            return test_data, test_labels
            
    def _load_SWAT(self, flag):
        """Load SWAT dataset"""
        train_data = pd.read_csv(os.path.join(self.root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(self.root_path, 'swat2.csv'))
        
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]
        
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        
        data_len = len(train_data)
        val_data = train_data[int(data_len * 0.8):]
        train_data = train_data[:int(data_len * 0.8)]
        
        if flag == 'train':
            return train_data, None
        elif flag == 'val':
            return val_data, None
        else:
            return test_data, labels
    
    def get_data(self, flag='train'):
        """Main method to get data based on dataset type"""
        if self.data_name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            return self.load_ETT_data(flag)
        elif self.data_name == 'custom':
            return self.load_custom_data(flag)
        elif self.data_name in ['PSM', 'MSL', 'SMAP', 'SMD', 'SWAT']:
            return self.load_anomaly_data(flag)
        else:
            # Try custom data loader
            return self.load_custom_data(flag)
            
    def inverse_transform(self, data):
        """Inverse transform scaled data"""
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data