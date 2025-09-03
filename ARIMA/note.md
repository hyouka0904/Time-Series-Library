# ARIMA框架設置說明

## 目錄結構

您的專案目錄結構應該如下：

```
專案根目錄/
├── ARIMA/                      # 您的ARIMA實驗框架（本目錄）
│   ├── arima_model.py
│   ├── arima_data_loader.py
│   ├── arima_experiments.py
│   ├── arima_utils.py
│   ├── run_arima_experiments.py
│   ├── compare_results.py
│   ├── run_all_arima_experiments.sh
│   └── README.md
├── dataset/                    # 數據集目錄（與原專案共用）
│   ├── ETT/
│   │   ├── ETTh1.csv
│   │   ├── ETTh2.csv
│   │   ├── ETTm1.csv
│   │   └── ETTm2.csv
│   ├── PSM/
│   ├── MSL/
│   ├── SMAP/
│   ├── SMD/
│   └── SWAT/
├── models/                     # 深度學習模型
├── exp/                        # 深度學習實驗
├── run.py                      # 深度學習主程式
└── ...

```

## 使用步驟

### 1. 確認數據集位置
確保您的數據集在`../dataset/`目錄下，這樣ARIMA框架才能正確讀取數據。

### 2. 運行ARIMA實驗
進入ARIMA目錄並執行實驗：

```bash
cd ARIMA
python run_arima_experiments.py --task_name long_term_forecast --data ETTh1 ...
```

### 3. 批次運行
使用提供的PowerShell腳本批次運行所有實驗：

```powershell
cd ARIMA
.\run_all_arima_experiments.ps1
```

如果遇到執行權限問題，請先執行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. 結果位置
- ARIMA結果會保存在`ARIMA/arima_results/`目錄
- 摘要結果會保存在ARIMA目錄下：
  - `result_long_term_forecast_arima.txt`
  - `result_anomaly_detection_arima.txt`
  - `result_imputation_arima.txt`

### 5. 比較結果
深度學習的結果應該在根目錄，ARIMA的結果在ARIMA目錄。執行比較腳本：

```bash
cd ARIMA
python compare_results.py
```

這會生成比較報告和視覺化圖表。

## 注意事項

1. **路徑問題**：所有路徑都已經配置為相對於ARIMA目錄，數據集使用`../dataset/`
2. **Python環境**：確保安裝了必要的套件（statsmodels, pmdarima等）
3. **運行時間**：ARIMA對長序列可能需要較長時間，特別是使用auto_arima時

## 快速測試

測試框架是否正常工作：

```bash
cd ARIMA

# 測試ETT數據集
python run_arima_experiments.py \
  --task_name long_term_forecast \
  --model_id quick_test \
  --data ETTh1 \
  --root_path ../dataset/ETT/ \
  --features S \
  --target OT \
  --seq_len 24 \
  --pred_len 24

# 測試Exchange Rate（如果有）
python run_arima_experiments.py \
  --task_name long_term_forecast \
  --model_id exchange_test \
  --data custom \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --features M \
  --seq_len 24 \
  --pred_len 24

# 測試M4數據集
python test_m4_arima.py
```

如果能成功運行並生成結果，說明設置正確。

## 運行完整實驗

### 運行所有基本實驗
```powershell
.\run_all_arima_experiments.ps1
```

### 運行Exchange Rate和M4實驗
```powershell
.\run_exchange_rate_and_m4.ps1
```
# Exchange Rate和M4數據集ARIMA實驗指南

## Exchange Rate數據集

### 數據準備
確認您的exchange_rate.csv格式如下：
```csv
date,currency1,currency2,currency3,...
2020-01-01,1.0,0.85,7.8,...
2020-01-02,1.01,0.86,7.85,...
```

第一列必須是`date`，其他列是不同貨幣的匯率。

### 運行實驗

#### 1. 多變量預測（預測所有貨幣）
```powershell
python run_arima_experiments.py --task_name long_term_forecast --model_id exchange_M --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features M --seq_len 96 --pred_len 96 --inverse
```

#### 2. 單變量預測（只預測一種貨幣，例如USD）
```powershell
python run_arima_experiments.py --task_name long_term_forecast --model_id exchange_USD --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features S --target USD --seq_len 96 --pred_len 96 --inverse
```

#### 3. 多變量預測單變量（使用所有貨幣預測USD）
```powershell
python run_arima_experiments.py --task_name long_term_forecast --model_id exchange_MS_USD --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features MS --target USD --seq_len 96 --pred_len 96 --inverse
```

### 注意事項
- 確保`--target`參數與CSV中的列名完全匹配
- Exchange rate數據通常有強相關性，M模式可能需要較長時間
- 建議先用較短的序列測試（如`--seq_len 24 --pred_len 24`）

## M4數據集

### 數據準備
M4數據集需要以下檔案：
```
../dataset/m4/
├── M4-info.csv      # 包含數據集元信息
├── training.npz     # 訓練數據
└── test.npz         # 測試數據
```

如果還沒有這些檔案，需要：
1. 從M4競賽官網下載原始數據
2. 使用data_provider中的腳本處理成npz格式

### 測試M4載入
運行測試腳本確認M4數據可以正確載入：
```powershell
cd ARIMA
python test_m4_arima.py
```

### 運行M4實驗

#### 單個季節性模式
```powershell
python run_arima_experiments.py --task_name short_term_forecast --model_id M4_Monthly --data m4 --root_path ../dataset/m4/ --seasonal_patterns Monthly
```

#### 所有季節性模式
```powershell
.\run_exchange_rate_and_m4.ps1
```

### M4季節性模式說明
- **Yearly**: 年度數據，預測6個時間點
- **Quarterly**: 季度數據，預測8個時間點  
- **Monthly**: 月度數據，預測18個時間點
- **Weekly**: 週數據，預測13個時間點
- **Daily**: 日數據，預測14個時間點
- **Hourly**: 小時數據，預測48個時間點

## 故障排除

### 1. ImportError: M4Dataset
如果出現M4Dataset導入錯誤：
- 確認`../data_provider/`目錄存在
- 確認`m4.py`檔案在該目錄中
- 檢查Python路徑設置

### 2. 數據未找到
- 確認數據路徑正確（相對於ARIMA目錄）
- 使用絕對路徑測試

### 3. ARIMA擬合失敗
某些時間序列可能無法擬合ARIMA模型：
- 數據太短
- 數據有太多零值
- 數據非平穩且差分後仍不平穩

程式會自動使用簡單預測方法作為備選。

### 4. 記憶體不足
對於大型數據集（如完整的M4）：
- 減少處理的序列數量（修改`_m4_experiment`中的限制）
- 使用更簡單的ARIMA參數（設置`auto_select=False`）
- 分批處理數據

## 效能優化建議

1. **並行處理**：可以修改代碼使用multiprocessing並行處理多個時間序列
2. **限制搜索範圍**：在auto_arima中設置較小的p、q範圍
3. **預先檢查**：跳過明顯不適合ARIMA的序列（如常數序列）

## 結果解讀

### Exchange Rate
- MSE/MAE：越小越好
- 多變量模式（M）會分別預測每個貨幣
- 注意匯率數據的尺度差異可能影響整體指標

### M4
- SMAPE：對稱平均絕對百分比誤差，越小越好
- OWA：整體加權平均，相對於基準方法的表現
- 不同季節性模式的SMAPE不能直接比較

# Exchange Rate數據集ARIMA實驗快速指南

## 基本命令

### 1. 長期預測（預測所有貨幣）
```powershell
cd ARIMA
python run_arima_experiments.py --task_name long_term_forecast --model_id exchange_96_96 --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features M --seq_len 96 --pred_len 96 --inverse
```

### 2. 短期預測
```powershell
python run_arima_experiments.py --task_name short_term_forecast --model_id exchange_24 --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features M --seq_len 48 --pred_len 24 --inverse
```

### 3. 異常檢測
```powershell
python run_arima_experiments.py --task_name anomaly_detection --model_id exchange_anomaly --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --anomaly_ratio 0.25
```

### 4. 插補
```powershell
python run_arima_experiments.py --task_name imputation --model_id exchange_impute --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features M --mask_rate 0.25
```

## 批次運行所有Exchange Rate和M4實驗

```powershell
.\run_exchange_rate_and_m4.ps1
```

## 常見問題

### Q: 如何只預測特定貨幣（如USD）？
A: 使用`--features S --target USD`：
```powershell
python run_arima_experiments.py --task_name long_term_forecast --model_id exchange_USD --data custom --root_path ../dataset/exchange_rate/ --data_path exchange_rate.csv --features S --target USD --seq_len 96 --pred_len 96 --inverse
```

### Q: 運行時間太長怎麼辦？
A: 
1. 減少序列長度：`--seq_len 24 --pred_len 24`
2. 只預測單一貨幣：`--features S --target USD`
3. 減少數據中的貨幣數量

### Q: 如何確認目標列名？
A: 查看CSV檔案的第一行，或運行：
```python
import pandas as pd
df = pd.read_csv('../dataset/exchange_rate/exchange_rate.csv')
print(df.columns.tolist())
```

## 預期結果
- 結果保存在：`./arima_results/`
- 摘要文件：`result_long_term_forecast_arima.txt`等
- 可與深度學習結果比較：`python compare_results.py`