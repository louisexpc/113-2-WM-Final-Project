## 🧠 Session Enrichment Pipeline 說明

本模組負責對使用者瀏覽紀錄進行語意強化（enrichment），透過大語言模型 (LLM) 預測使用者偏好商品類別，並加入適當的類別資訊，使得序列更具語意與預測性。

---

## ⚙️ 設定檔說明 (`config/enrich_sessions.yaml`)

以下為 `enrich_sessions.yaml` 中各參數區塊的用途與說明：

### 🔹 model 模型相關設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `name` | Hugging Face 上的大語言模型名稱 | `"meta-llama/Llama-3.1-8b-Instruct"` |
| `top_k` | 為每位使用者挑選相似的前 k 個商品類別 | `20` |
| `total_len` | enrichment 後希望的 session 長度（例如填滿到 30）| `30` |
| `gpu_memory_utilization` | 使用 GPU 記憶體比例 (0~1)，vLLM 設定 | `0.8` |
| `max_model_len` | 最大 token 長度，依模型大小調整 | `2048` |
| `max_num_seqs` | vLLM 批次處理序列數上限 | `16` |


---

### 🔹 short_model 分類用嵌入模型設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `name` | Sentence-BERT 模型，用來生成商品類別向量 | `"all-MiniLM-L6-v2"` |

---

### 🔹 data 資料來源設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `session_path` | 使用者瀏覽紀錄 `.pkl` 檔 | `"dataset/sessions_5_6_30_mapping.pkl"` |
| `category_csv` | 所有商品類別列表檔案（含欄位 `category_name`）| `"dataset/product_types.csv"` |
| `prompt_path` | 提供給 LLM 的 prompt 模板（.txt）| `"prompts/get_product_type_from_history_prompt_v1.1.txt"` |
| `mapping_path` | 商品與類別對應檔案，用於 query 時類別過濾 | `"dataset/articles_mapping.csv"` |

---

### 🔹 save 輸出儲存設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `checkpoint_path` | 中途儲存的檔案路徑（可 resume）| `"dataset/enriched_sessions_checkpoint.pkl"` |
| `final_output_path` | 所有使用者處理完後的輸出結果 | `"dataset/enriched_sessions_final.pkl"` |
| `batch_size` | 每處理幾筆資料即儲存 checkpoint | `5` |
| `max_users` | 最多處理幾位使用者（用於測試或限制規模）| `10` |

---

### 🔹 device GPU 裝置設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `cuda_visible_devices` | 可用的 GPU 裝置 ID（字串形式） | `"4"` |

---

### 🔹 experiment 實驗輸出設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `id` | 實驗名稱或編號，用於輸出資料夾命名 | `"enrich_0526_llama_v1"` |
| `output_base_dir` | 所有輸出的根目錄 | `"./experiments/enrich_sessions/"` |
| `create_timestamp_dir` | 是否在 output 資料夾後自動加上 timestamp 子資料夾 | `true` |

---

### 🔹 hf Hugging Face 認證設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `use_token` | 是否使用 Hugging Face token 進行 API 認證 | `true` |
| `token_env_key` | 存放 token 的環境變數名稱 | `"HF_TOKEN"` |

需要在`./session_generation` 下新增 `.env `:
```
HF_TOKEN = <your token here>
```

---

## 📝 輸出結果格式

最終會產出一個 `.pkl` 檔，其格式為：

```python
{
  "user_id_1": (["category_a", "category_b", ...], k1),
  "user_id_2": (["category_c", "category_d", ...], k2),
  ...
}
```

## 🛍️ Product Name Generation Pipeline 說明

本模組根據強化後的使用者 session，透過大語言模型（LLM）與用戶個資條件，自動生成具有創意且個人化的商品名稱。最終輸出對應每位使用者的推薦商品名稱清單。

---

## ⚙️ 設定檔說明 (`config/gen_product_names.yaml`)

以下為 `gen_product_names.yaml` 中各參數區塊的用途與說明：

### 🔹 experiment 實驗設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `id` | 實驗名稱或代碼，用於輸出資料夾命名 | `"gen_product_names_llama3"` |
| `output_base_dir` | 所有輸出的根目錄 | `"./experiments/gen_product_names/"` |
| `create_timestamp_dir` | 是否在 output 資料夾後自動加上 timestamp 子資料夾 | `true` |

---

### 🔹 device GPU 裝置設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `cuda_visible_devices` | 可用的 GPU 裝置 ID（字串形式） | `"4"` |

---

### 🔹 model 模型設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `name` | Hugging Face 上的 LLM 模型名稱 | `"meta-llama/Llama-3.1-8b-Instruct"` |
| `local_model_dir` | 模型本地快取路徑（若已下載） | `"./llama3_hf_cache"` |
| `gpu_memory_utilization` | GPU 記憶體使用率（vLLM 參數） | `0.8` |
| `max_model_len` | 模型最大 token 長度 | `2048` |
| `max_num_seqs` | 同時處理的序列數上限（vLLM 批次參數） | `16` |
| `temperature` | 控制生成隨機性（越低越保守） | `0.3` |
| `top_p` | nucleus sampling 機率門檻 | `0.9` |
| `max_tokens` | 每次生成的最大 token 數 | `128` |

---

### 🔹 data 資料來源與儲存設定

| 參數 | 說明 | 範例 |
|------|------|------|
| `input_path` | 已完成 enrichment 的使用者 session 資料 | `"dataset/enriched_sessions_final.pkl"` |
| `customers_path` | 使用者屬性檔（包含年齡、會員狀態、時尚新聞頻率等）| `"dataset/customers_mapping.csv"` |
| `output_path` | 生成後的商品名稱輸出檔案路徑 | `"dataset/generated_product_name_4.pkl"` |
| `product_examples_path` | 類別對應的商品命名範例 CSV 檔，用於提供 prompt | `"dataset/product_name_examples.csv"` |
| `batch_size` | 每批處理的使用者數（用於記憶體控管） | `5` |

---

## 📝 輸出結果格式

本模組會產出一個 `.pkl` 檔案，格式如下：

```python
{
  "user_id_1": ["Knit Jacket", "Linen Dress", ...],
  "user_id_2": ["Slim Jeans", "Silk Blazer", ...],
  ...
}
