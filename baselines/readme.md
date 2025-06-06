# Neural Collaborative Filtering (NCF) - train_ncf.py

## 執行需求
- Python 3.8+
- PyTorch, tqdm, numpy

## 資料結構
請確認下列檔案已置於 `./dataset/filtered-h-and-m/processed_data/`：
- `user_session.pkl`
- `testing_data.pkl`

## 執行方式
```bash
python train_ncf.py --output_csv_path result/ncf_result.csv
```
---

# Bayesian Personalized Ranking (BPR) - train_bpr.py

## 執行需求
- Python 3.8+
- PyTorch, tqdm, numpy

## 資料結構
請確認下列檔案已置於 `./dataset/filtered-h-and-m/processed_data/`：
- `user_session.pkl`
- `testing_data.pkl`

## 執行方式
```bash
python train_bpr.py --output_csv_path result/bpr_result.csv
```
