import argparse
import os
import numpy as np
import random
from exp import (
    ARIMALongTermForecast, 
    ARIMAAnomalyDetection,
    ARIMAImputation
)


def set_seed(seed=2021):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    """Print experiment arguments"""
    print("="*50)
    print("Experiment Arguments:")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='ARIMA Experiments')
    
    # Basic config
    parser.add_argument('--task_name', type=str, required=True, 
                        default='long_term_forecast',
                        help='task name: long_term_forecast, anomaly_detection, imputation')
    parser.add_argument('--model_id', type=str, required=True, 
                        default='test', help='model id')
    
    # Data loader
    parser.add_argument('--data', type=str, required=True, 
                        default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, 
                        default='./data/ETT/', help='root path of data file')
    parser.add_argument('--data_path', type=str, 
                        default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task: M, S, MS')
    parser.add_argument('--target', type=str, default='OT', 
                        help='target feature in S or MS task')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, 
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, 
                        help='start token length (not used in ARIMA)')
    parser.add_argument('--pred_len', type=int, default=96, 
                        help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', 
                        help='inverse output data', default=False)
    
    # Imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, 
                        help='mask ratio')
    
    # Anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, 
                        help='prior anomaly ratio (%)')
    
    # Model parameters (ARIMA specific)
    parser.add_argument('--auto_arima', action='store_true', 
                        default=True, help='use auto ARIMA')
    parser.add_argument('--arima_p', type=int, default=1, 
                        help='ARIMA p parameter')
    parser.add_argument('--arima_d', type=int, default=1, 
                        help='ARIMA d parameter')
    parser.add_argument('--arima_q', type=int, default=1, 
                        help='ARIMA q parameter')
    
    # Others
    parser.add_argument('--itr', type=int, default=1, 
                        help='experiments times')
    parser.add_argument('--des', type=str, default='test', 
                        help='exp description')
    
    args = parser.parse_args()
    
    # Print arguments
    print_args(args)
    
    # Set random seed
    set_seed(2021)
    
    # Select experiment based on task
    if args.task_name == 'long_term_forecast':
        Exp = ARIMALongTermForecast
    elif args.task_name == 'anomaly_detection':
        Exp = ARIMAAnomalyDetection
    elif args.task_name == 'imputation':
        Exp = ARIMAImputation
    else:
        raise ValueError(f"Unknown task: {args.task_name}")
    
    # Run experiments
    for ii in range(args.itr):
        # Create experiment setting string
        setting = f'{args.task_name}_{args.model_id}_ARIMA_{args.data}_' \
                 f'ft{args.features}_sl{args.seq_len}_pl{args.pred_len}_' \
                 f'{args.des}_{ii}'
        
        print(f'\n>>>>>>>start experiment : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        
        # Initialize and run experiment
        exp = Exp(args)
        
        # Train and test
        if args.task_name == 'long_term_forecast':
            mse, mae, rmse = exp.train_and_test(setting)
        elif args.task_name == 'anomaly_detection':
            accuracy, precision, recall, f_score = exp.train_and_test(setting)
        elif args.task_name == 'imputation':
            mse, mae, rmse = exp.train_and_test(setting)
        
        print(f'>>>>>>>experiment complete : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')


if __name__ == '__main__':
    main()