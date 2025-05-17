# 113-2-WM-Final-Project

# Dataset - H&M Kaggel
- [link of H&M Kaggle dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

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
## Classification of Data
| Step | Description                                                            | Output Filename(s)                            |
|------|------------------------------------------------------------------------|-----------------------------------------------|
| 1    | Remove articles without `detail_desc` feature                          | `transaction_train_clean.csv`                 |
| 2    | Remove articles with fewer than 5 transaction records (Cold Start)     | `transaction_5.csv`                           |
| 3    | Remove transactions older than 24 weeks for each customer              | —                                             |
| 4    | Remove customers with fewer than 4 or 6 transactions                   | `transaction_5_4.csv` / `transaction_5_6.csv` |
| 5    | Turn transaction record into session-based data                        | `session_5_4.pkl` / `session_5_6.pkl`         |
| 6    | Transform session data into baseline dataset (optional)               | `user_session_5_4.pkl` / `testing_data_5_4.pkl`<br>`user_session_5_6.pkl` / `testing_data_5_6.pkl` |

## Quick Start
- For `transaction_....csv`
    ```python
    # Default data type for each cols
    trans = pd.read_csv(r"C:\113-2-WM-Final-Project\data\transactions_train.csv",
                 parse_dates=['t_dat'],
                 dtype={
                     'customer_id':'category',
                     'article_id': 'int32',
                     'sales_channel_id':'uint8'
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
            'article_id'      : [int32, ...],
            't_dat'           : [Timestamp, ...],
            'price'           : [float32, ...],
            'sales_channel_id': [uint8, ...]
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
