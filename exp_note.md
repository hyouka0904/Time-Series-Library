# ARIMA
## exchange_rate
python run.py --task_name long_term_forecast --model_id exchange_rate_96_96 --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features M --seq_len 96 --pred_len 96 --des Exp --inverse

python run.py --task_name long_term_forecast --model_id exchange_rate_96_192 --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features M --seq_len 96 --pred_len 192 --des Exp --inverse

python run.py --task_name long_term_forecast --model_id exchange_rate_96_336 --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features M --seq_len 96 --pred_len 336 --des Exp --inverse

python run.py --task_name long_term_forecast --model_id exchange_rate_96_720 --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features M --seq_len 96 --pred_len 720 --des Exp --inverse
## weather

python run.py --task_name long_term_forecast --model_id weather_96_96 --data custom --root_path ../dataset/weather/ --data_path weather.csv --features M --seq_len 96 --pred_len 96 --des Exp --inverse

python run.py --task_name long_term_forecast --model_id weather_96_192 --data custom --root_path ../dataset/weather/ --data_path weather.csv --features M --seq_len 96 --pred_len 192 --des Exp --inverse

python run.py --task_name long_term_forecast --model_id weather_96_336 --data custom --root_path ../dataset/weather/ --data_path weather.csv --features M --seq_len 96 --pred_len 336 --des Exp --inverse

python run.py --task_name long_term_forecast --model_id weather_96_720 --data custom --root_path ../dataset/weather/ --data_path weather.csv --features M --seq_len 96 --pred_len 720 --des Exp --inverse
# TimesNet
## exchange_rate
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_96_96 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 64 --top_k 5 --des 'Exp' --itr 1
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_96_192 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 64 --top_k 5 --des 'Exp' --itr 1
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_96_336 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --d_model 32 --d_ff 32 --top_k 5 --des 'Exp' --itr 1 --train_epochs 1
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_96_720 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --d_model 32 --d_ff 32 --top_k 5 --des 'Exp' --itr 1 --train_epochs 1

## weather
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_96 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --d_model 32 --d_ff 32 --top_k 5 --des 'Exp' --itr 1
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_192 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --d_model 32 --d_ff 32 --top_k 5 --des 'Exp' --itr 1 --train_epochs 1
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_336 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --d_model 32 --d_ff 32 --top_k 5 --des 'Exp' --itr 1
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_720 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --d_model 32 --d_ff 32 --top_k 5 --des 'Exp' --itr 1 --train_epochs 1

# TSMixer
## exchange_rate
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_96_96 --model TSMixer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_96_192 --model TSMixer --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_96_336 --model TSMixer --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1
$env:CUDA_VISIBLE_DEVICES = "0" ; python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_96_720 --model TSMixer --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1

