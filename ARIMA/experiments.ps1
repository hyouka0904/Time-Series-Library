# ARIMA Experiments PowerShell Script
# Each line runs a single experiment

Write-Host "Starting ARIMA experiments..." -ForegroundColor Green

# Long-term Forecasting Experiments
Write-Host "`n=== Long-term Forecasting Experiments ===" -ForegroundColor Yellow

# ETTh1 - prediction length 96
python run.py --task_name long_term_forecast --model_id ETTh1_96_96 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --seq_len 96 --pred_len 96 --des Exp --inverse

# ETTh1 - prediction length 192
python run.py --task_name long_term_forecast --model_id ETTh1_96_192 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --seq_len 96 --pred_len 192 --des Exp --inverse

# ETTh1 - prediction length 336
python run.py --task_name long_term_forecast --model_id ETTh1_96_336 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --seq_len 96 --pred_len 336 --des Exp --inverse

# ETTh1 - prediction length 720
python run.py --task_name long_term_forecast --model_id ETTh1_96_720 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --seq_len 96 --pred_len 720 --des Exp --inverse

# ETTh2 - prediction length 96
python run.py --task_name long_term_forecast --model_id ETTh2_96_96 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --seq_len 96 --pred_len 96 --des Exp --inverse

# ETTh2 - prediction length 192
python run.py --task_name long_term_forecast --model_id ETTh2_96_192 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --seq_len 96 --pred_len 192 --des Exp --inverse

# ETTh2 - prediction length 336
python run.py --task_name long_term_forecast --model_id ETTh2_96_336 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --seq_len 96 --pred_len 336 --des Exp --inverse

# ETTh2 - prediction length 720
python run.py --task_name long_term_forecast --model_id ETTh2_96_720 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --seq_len 96 --pred_len 720 --des Exp --inverse

# ETTm1 - prediction length 96
python run.py --task_name long_term_forecast --model_id ETTm1_96_96 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --seq_len 96 --pred_len 96 --des Exp --inverse

# ETTm1 - prediction length 192
python run.py --task_name long_term_forecast --model_id ETTm1_96_192 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --seq_len 96 --pred_len 192 --des Exp --inverse

# ETTm1 - prediction length 336
python run.py --task_name long_term_forecast --model_id ETTm1_96_336 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --seq_len 96 --pred_len 336 --des Exp --inverse

# ETTm1 - prediction length 720
python run.py --task_name long_term_forecast --model_id ETTm1_96_720 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --seq_len 96 --pred_len 720 --des Exp --inverse

# ETTm2 - prediction length 96
python run.py --task_name long_term_forecast --model_id ETTm2_96_96 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --seq_len 96 --pred_len 96 --des Exp --inverse

# ETTm2 - prediction length 192
python run.py --task_name long_term_forecast --model_id ETTm2_96_192 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --seq_len 96 --pred_len 192 --des Exp --inverse

# ETTm2 - prediction length 336
python run.py --task_name long_term_forecast --model_id ETTm2_96_336 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --seq_len 96 --pred_len 336 --des Exp --inverse

# ETTm2 - prediction length 720
python run.py --task_name long_term_forecast --model_id ETTm2_96_720 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --seq_len 96 --pred_len 720 --des Exp --inverse

# Short-term Forecasting Experiments
Write-Host "`n=== Short-term Forecasting Experiments ===" -ForegroundColor Yellow

# ETTh1 short-term - prediction length 24
python run.py --task_name short_term_forecast --model_id ETTh1_short_24 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --seq_len 48 --pred_len 24 --des Exp --inverse

# ETTh1 short-term - prediction length 48
python run.py --task_name short_term_forecast --model_id ETTh1_short_48 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --seq_len 96 --pred_len 48 --des Exp --inverse

# ETTh2 short-term - prediction length 24
python run.py --task_name short_term_forecast --model_id ETTh2_short_24 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --seq_len 48 --pred_len 24 --des Exp --inverse

# ETTh2 short-term - prediction length 48
python run.py --task_name short_term_forecast --model_id ETTh2_short_48 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --seq_len 96 --pred_len 48 --des Exp --inverse

# ETTm1 short-term - prediction length 24
python run.py --task_name short_term_forecast --model_id ETTm1_short_24 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --seq_len 48 --pred_len 24 --des Exp --inverse

# ETTm1 short-term - prediction length 48
python run.py --task_name short_term_forecast --model_id ETTm1_short_48 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --seq_len 96 --pred_len 48 --des Exp --inverse

# ETTm2 short-term - prediction length 24
python run.py --task_name short_term_forecast --model_id ETTm2_short_24 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --seq_len 48 --pred_len 24 --des Exp --inverse

# ETTm2 short-term - prediction length 48
python run.py --task_name short_term_forecast --model_id ETTm2_short_48 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --seq_len 96 --pred_len 48 --des Exp --inverse

# M4 dataset experiments
python run.py --task_name short_term_forecast --model_id M4_Monthly --data m4 --root_path ../dataset/m4/ --seasonal_patterns Monthly --des Exp
python run.py --task_name short_term_forecast --model_id M4_Yearly --data m4 --root_path ../dataset/m4/ --seasonal_patterns Yearly --des Exp
python run.py --task_name short_term_forecast --model_id M4_Quarterly --data m4 --root_path ../dataset/m4/ --seasonal_patterns Quarterly --des Exp
python run.py --task_name short_term_forecast --model_id M4_Weekly --data m4 --root_path ../dataset/m4/ --seasonal_patterns Weekly --des Exp
python run.py --task_name short_term_forecast --model_id M4_Daily --data m4 --root_path ../dataset/m4/ --seasonal_patterns Daily --des Exp
python run.py --task_name short_term_forecast --model_id M4_Hourly --data m4 --root_path ../dataset/m4/ --seasonal_patterns Hourly --des Exp

# Anomaly Detection Experiments
Write-Host "`n=== Anomaly Detection Experiments ===" -ForegroundColor Yellow

# PSM dataset
python run.py --task_name anomaly_detection --model_id PSM --data PSM --root_path ../dataset/PSM/ --anomaly_ratio 0.25 --des Exp

# MSL dataset
python run.py --task_name anomaly_detection --model_id MSL --data MSL --root_path ../dataset/MSL/ --anomaly_ratio 0.25 --des Exp

# SMAP dataset
python run.py --task_name anomaly_detection --model_id SMAP --data SMAP --root_path ../dataset/SMAP/ --anomaly_ratio 0.25 --des Exp

# SMD dataset
python run.py --task_name anomaly_detection --model_id SMD --data SMD --root_path ../dataset/SMD/ --anomaly_ratio 0.25 --des Exp

# SWAT dataset
python run.py --task_name anomaly_detection --model_id SWAT --data SWAT --root_path ../dataset/SWAT/ --anomaly_ratio 0.25 --des Exp

# Imputation Experiments
Write-Host "`n=== Imputation Experiments ===" -ForegroundColor Yellow

# ETTh1 - mask rate 0.125
python run.py --task_name imputation --model_id ETTh1_mask_0.125 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --mask_rate 0.125 --des Exp

# ETTh1 - mask rate 0.25
python run.py --task_name imputation --model_id ETTh1_mask_0.25 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --mask_rate 0.25 --des Exp

# ETTh1 - mask rate 0.375
python run.py --task_name imputation --model_id ETTh1_mask_0.375 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --mask_rate 0.375 --des Exp

# ETTh1 - mask rate 0.5
python run.py --task_name imputation --model_id ETTh1_mask_0.5 --data ETTh1 --root_path ../dataset/ETT/ --data_path ETTh1.csv --features M --mask_rate 0.5 --des Exp

# ETTh2 - mask rate 0.125
python run.py --task_name imputation --model_id ETTh2_mask_0.125 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --mask_rate 0.125 --des Exp

# ETTh2 - mask rate 0.25
python run.py --task_name imputation --model_id ETTh2_mask_0.25 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --mask_rate 0.25 --des Exp

# ETTh2 - mask rate 0.375
python run.py --task_name imputation --model_id ETTh2_mask_0.375 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --mask_rate 0.375 --des Exp

# ETTh2 - mask rate 0.5
python run.py --task_name imputation --model_id ETTh2_mask_0.5 --data ETTh2 --root_path ../dataset/ETT/ --data_path ETTh2.csv --features M --mask_rate 0.5 --des Exp

# ETTm1 - mask rate 0.125
python run.py --task_name imputation --model_id ETTm1_mask_0.125 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --mask_rate 0.125 --des Exp

# ETTm1 - mask rate 0.25
python run.py --task_name imputation --model_id ETTm1_mask_0.25 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --mask_rate 0.25 --des Exp

# ETTm1 - mask rate 0.375
python run.py --task_name imputation --model_id ETTm1_mask_0.375 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --mask_rate 0.375 --des Exp

# ETTm1 - mask rate 0.5
python run.py --task_name imputation --model_id ETTm1_mask_0.5 --data ETTm1 --root_path ../dataset/ETT/ --data_path ETTm1.csv --features M --mask_rate 0.5 --des Exp

# ETTm2 - mask rate 0.125
python run.py --task_name imputation --model_id ETTm2_mask_0.125 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --mask_rate 0.125 --des Exp

# ETTm2 - mask rate 0.25
python run.py --task_name imputation --model_id ETTm2_mask_0.25 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --mask_rate 0.25 --des Exp

# ETTm2 - mask rate 0.375
python run.py --task_name imputation --model_id ETTm2_mask_0.375 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --mask_rate 0.375 --des Exp

# ETTm2 - mask rate 0.5
python run.py --task_name imputation --model_id ETTm2_mask_0.5 --data ETTm2 --root_path ../dataset/ETT/ --data_path ETTm2.csv --features M --mask_rate 0.5 --des Exp

Write-Host "`nAll ARIMA experiments completed!" -ForegroundColor Green