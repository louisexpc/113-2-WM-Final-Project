# 113-2-WM-Final-Project

## Dataset - H&M Kaggel
- [link of H&M Kaggle dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)


# Project Structure
```
.
├── baselines/
│   ├── ana_bpr.py
│   ├── bpr.py
│   ├── ncf.py
│   ├── readme.md         # <-- 基準模型的執行說明
│   ├── train_bpr.py
│   └── train_ncf.py
│
├── data_preprocessing/
│   ├── preprocess_transaction.ipynb
│   └── sessions_transform.ipynb
│
├── eda/
│   ├── articles_eda.ipynb
│   └── session_eda.ipynb
│
├── src/
│   ├── config/
│   ├── dataset/
│   ├── modules/
│   ├── prompts/
│   ├── utils/
│   ├── README.md         # <-- 主要 LLM 流程的執行說明
│   ├── requirements.txt
│   ├── run.py
│   └── run.sh
│
└── README.md             # <-- 您正在閱讀的這份總說明檔案
```
---
## 各目錄功能說明
- `src`  
這是本專案的核心部分，包含一個完整的多階段 LLM Session Augmentation 流程，可參考下方流程圖。它會處理使用者會話 (Session)，生成新的產品概念，並將其對應到真實世界的商品。若您想執行主要的 LLM Augmentation流程，請參考此目錄下的說明。
> ![](./eda/outputs/A4%20-%201.jpg) 

- `baselines`  
這個目錄存放了數個傳統但有效的推薦演算法，例如 BPR (Bayesian Personalized Ranking) 和 NCF (Neural Collaborative Filtering)。這些模型被用來作為比較 src 中 Augmentation 效能的基準線。若您想訓練或評估這些經典模型，請參考此目錄下的說明。

- `data_preprocessing`  
此目錄包含對 H&M 資料集進行前處理的 Jupyter Notebooks。主要功能是將原始交易資料轉換為模型可用的會話格式 (session-based format)。這是所有模型（包含 `src` 和 `baselines`）執行前的第一步

- `eda`  
此目錄存放用於探索性資料分析的 Jupyter Notebooks。透過分析商品 (`articles`) 和使用者會話 (`session`) 的特性，來獲取對資料的洞察，並作為模型設計的依據。

## 如何開始 (Getting Started)
根據您的目標，請參考對應的 `README.md` 檔案來執行：

### 🚀 執行主要的 LLM Session Augmentation 流程
1. 進入 src 目錄：
    ```
    cd src
    ```
    詳細閱讀並遵循 `src/README.md` 中的指引來安裝依賴套件、設定環境並執行完整的步驟。

### 📊 執行基準模型 (BPR/NCF)
1. 
    ```
    cd baselines
    ```
    詳細閱讀並遵循 `baselines/readme.md` 中的指引來安裝依賴套件、設定環境並執行完整的模型訓練與比較。