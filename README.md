# 113-2-WM-Final-Project

# Dataset - H&M Kaggel
- [link of H&M Kaggle dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

# Generated_imgs
- [link](https://drive.google.com/drive/folders/1f3mHljg80oDsPfm3zyTsnXIiZzbEsuY3?usp=drive_link)

# Project Structure
```
- data: only for small data
- eda: 
   - output: pics of eda_result.md
- preprocessing: 
- src
```

# Data 使用方式
- [Data Space](https://drive.google.com/drive/folders/15yY3Y58dTSp_yLDWK5TkhHQZEE-My069?usp=sharing)
- Data Folder Structure
    ```
    ├── original data/ : Kaggle Origial Data
    ├── mapping data/  : Data with id mapping and preprocessing
    ├── baseline data/ : Data for baseline model (without id mapping)
    ```
## Classification of `mapping data`
- Mapping Dict:
    | Filename | Description                                                               |
    |----------|---------------------------------------------------------------------------|
    | `customer_to_idx.pkl`               | `Dict[ orgin customer_id : int ]`              | 
    | `idx_to_customer.pkl`               | `Dict[ int : orgin customer_id ]`               |
    | `article_to_idx.pkl`                | `Dict[ orgin article_id : int ]`               | 
    | `idx_to_article.pkl`                | `Dict[ int : orgin article_i ]`                | 
- Data without preprocessing:
    | Filename | Description                                                               |
    |----------|---------------------------------------------------------------------------|
    | `transaction_train_mapping.csv`       | 將原始 `customer_id`,`article_id` 進行 mapping| 
    | `articles_mapping.csv`                | 將原始 `article_id` 進行 mapping              |
    | `customer_mapping.csv`                | 將原始 `customer_id` 進行 mapping             | 
                         
- Data after preprocessing: 
    | Step | Description                                                            | Output Filename(s)       |
    |------|------------------------------------------------------------------------|-----------------------------------------------|
    | 1    | Remove articles without `detail_desc` feature                          | `transaction_train_mapping_clean.csv`         |
    | 2    | Remove articles with fewer than 5 transaction records (Cold Start)     | `transaction_5_mapping.csv`                   |
    | 3    | Remove transactions older than 24 weeks for each customer              | —                                             |
    | 4    | Remove customers with fewer than `4` or `6` transactions or more than `30` transactions| `transaction_5_4_30_mapping.csv` / `transaction_5_6_30_mapping.csv` |
    | 5    | Turn transaction record into session-based data                        | `session_5_4_30_mapping.pkl` / `session_5_6_30_mapping.pkl`         |


## Quick Start
- For `transaction_....csv`
    ```python
    # Default data type for each cols
    trans = pd.read_csv(r"C:\113-2-WM-Final-Project\data\transactions_train.csv",
                 parse_dates=['t_dat'],
                 dtype={
                     'customer_id':'int',
                     'article_id': 'int',
                     'price': 'float'
                     'sales_channel_id':'int'
                 })
    ```
- For `*.pkl`:
    ```python
    import pickle
    with open(r"your_file.pkl", "rb") as f:
        file = pickle.load(f)
    ```
## Data Format
- For `session_...` : `Dict[str, Dict[str, list]]`
    ```
    sessions : {
        customer_id: {
            'article_id'      : [int, ...],
            't_dat'           : [Timestamp, ...],
            'price'           : [float32, ...],
            'sales_channel_id': [int, ...]
        },
        ...
     }
    ```
- For baseline dataset: `Dict[str, List[int]]`
    ```
    -   user_session : Dict[str, List[int]]
            { uid: [iid_1, iid_2, ..., iid_n] }
    -   testing_data : Dict[str, List[int]]
            { uid: [neg_1, ..., neg_99, test_item] }
    ```
