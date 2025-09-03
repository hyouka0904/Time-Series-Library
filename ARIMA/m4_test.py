"""
Test script for M4 dataset with ARIMA
Run this to verify M4 dataset loading works correctly
"""
import sys
import os
sys.path.append('..')  # Add parent directory to path

from data_loader import ARIMADataLoader
from model import ARIMAModel
import numpy as np
import pandas as pd


def test_m4_loading():
    """Test if M4 dataset can be loaded correctly"""
    print("Testing M4 dataset loading...")
    
    # Test different seasonal patterns
    seasonal_patterns = ['Monthly', 'Yearly', 'Quarterly']
    
    for pattern in seasonal_patterns:
        print(f"\nTesting {pattern} pattern...")
        
        try:
            # Create data loader
            loader = ARIMADataLoader(
                data_name='m4',
                root_path='../dataset/m4/',
                seasonal_patterns=pattern
            )
            
            # Load training data
            train_data, train_ids = loader.get_data('train')
            print(f"  Loaded {len(train_data)} training series")
            print(f"  First series length: {len(train_data[0])}")
            print(f"  Sample IDs: {train_ids[:5]}")
            
            # Load test data
            test_info, test_ids = loader.get_data('test')
            if isinstance(test_info, tuple):
                train_context, test_data = test_info
                print(f"  Loaded {len(test_data)} test series")
            
            # Test ARIMA on first series
            print(f"\n  Testing ARIMA on first {pattern} series...")
            model = ARIMAModel(auto_select=True)
            model.fit(train_data[0], normalize=False)
            
            # Make prediction
            from data_provider.m4 import M4Meta
            pred_len = M4Meta.horizons_map[pattern]
            predictions = model.predict(pred_len)
            print(f"  Made {len(predictions)} predictions")
            print(f"  Predictions: {predictions[:5]}")
            
            print(f"\n  {pattern} test passed!")
            
        except Exception as e:
            print(f"\n  Error testing {pattern}: {e}")
            import traceback
            traceback.print_exc()


def test_m4_experiment():
    """Test running a complete M4 experiment"""
    print("\n\nTesting complete M4 experiment...")
    
    # Create mock args
    class Args:
        task_name = 'short_term_forecast'
        model_id = 'M4_test'
        data = 'm4'
        root_path = '../dataset/m4/'
        seasonal_patterns = 'Monthly'
        des = 'Test'
        # These will be set by the experiment
        pred_len = None
        seq_len = None
        label_len = None
        frequency_map = None
    
    args = Args()
    
    try:
        from exp import ARIMAShortTermForecast
        exp = ARIMAShortTermForecast(args)
        
        setting = f'{args.task_name}_{args.model_id}_{args.data}_{args.seasonal_patterns}'
        smape, owa = exp.train_and_test(setting)
        
        print(f"\nExperiment completed!")
        print(f"SMAPE: {smape:.4f}")
        print(f"OWA: {owa:.4f}")
        
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("M4 Dataset ARIMA Test")
    print("=" * 50)
    
    # Check if M4 dataset exists
    m4_path = '../dataset/m4/'
    if not os.path.exists(m4_path):
        print(f"Error: M4 dataset not found at {m4_path}")
        print("Please download and extract M4 dataset to this location")
        print("\nExpected structure:")
        print(f"{m4_path}")
        print("├── M4-info.csv")
        print("├── training.npz")
        print("└── test.npz")
    else:
        # Run tests
        test_m4_loading()
        test_m4_experiment()
        
    print("\n" + "=" * 50)
    print("Test completed!")