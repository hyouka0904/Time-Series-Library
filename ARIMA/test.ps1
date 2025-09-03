
python run.py --task_name long_term_forecast --model_id weather_96_96 --data custom --root_path ../dataset/weather/ --data_path weather.csv --features M --seq_len 96 --pred_len 96 --des Exp --inverse

python run.py --task_name long_term_forecast --model_id weather_96_192 --data custom --root_path ../dataset/weather/ --data_path weather.csv --features M --seq_len 96 --pred_len 192 --des Exp --inverse

python run.py --task_name long_term_forecast --model_id weather_96_336 --data custom --root_path ../dataset/weather/ --data_path weather.csv --features M --seq_len 96 --pred_len 336 --des Exp --inverse

python run.py --task_name long_term_forecast --model_id weather_96_720 --data custom --root_path ../dataset/weather/ --data_path weather.csv --features M --seq_len 96 --pred_len 720 --des Exp --inverse