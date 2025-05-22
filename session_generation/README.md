# User Session Generator

這是一份用於「冷啟動推薦系統」模擬的實驗原型，我們的目標是針對購買紀錄較短的使用者，透過語意分析與大型語言模型（LLM）產生擬真的購買序列，補齊完整的 user session。

---

## ✅ 目前完成的功能

### 1. 使用者 session 擴充流程

對每位使用者，我們會根據其已購商品 ID：

1. 將 `article_id` 轉換成對應的 `product_type`
2. 將所有商品類別進行語意比對，選出前 20 相似的類別（使用 sentence-transformers 模型）
3. 將這些類別與使用者的購買紀錄一併輸入至 LLM，預測可能會購買的 n 個商品類別
4. 對每個預測出的商品類別，再呼叫一次 LLM 生成對應的商品名稱
5. 將這些名稱插入使用者原本的購買紀錄中（預設插入倒數第二個位置）

### 2. 核心模組簡介

| 函數名稱 | 說明 |
|----------|------|
| `prefilter_categories()` | 根據使用者歷史商品，從所有類別中篩選出 top-K 類別（避免爆 token） |
| `get_categories_from_history()` | 用 LLM 根據使用者紀錄選出可能會購買的 n 個類別 |
| `get_item_from_category()` | 根據類別由 LLM 產生擬真的商品名稱 |
| `extract_list_from_text()` / `extract_product_name()` | 將 LLM 輸出中提取合法格式資料（例如 list 或產品名稱） |
| `enrich_user_session()` | 主流程函數，串起整體的處理與擴充邏輯 |

---

## 📦 輸入資料格式範例

```python
user_session = {
    "user_1": [841260003, 887593002, 890498002],
    "user_2": [759191008, 800436010]
}
```

---

## 🔁 輸出資料格式範例（enriched）

```python
{
  'user_1': [841260003, 887593002, 'FlexFit Vest', 'Comfy Breeze Tee', 890498002],
  'user_2': [759191008, 'Wave Crusher Top', 'SunChic Swim Set', 800436010]
}
```

---

## 🧪 使用的模型與工具

- LLM 模型：`meta-llama/Llama-2-7b-chat-hf`（需申請 access token）
- 向量語意模型：`sentence-transformers/all-MiniLM-L6-v2`
- GPU 環境：需支援 CUDA 的顯示卡（RTX 4090 最佳）

---

## ⏳ 範例執行時間

- 測試資料：每個使用者有約 6 筆紀錄
- user_session 共 2 筆
- 總執行時間：約 20 秒
- 產出商品名稱共 6 個

---

## 📌 待辦事項（To Do）

- 商品名稱生成品質不穩，考慮建立真實商品名稱池或微調模型
- 提高 LLM 的控制力，例如避免重複、簡化 prompt
- 增加完整商品資訊（如描述、價格等）進入 session
- 優化 runtime：batch 處理、async 推理等
- 加入錯誤處理與日誌紀錄

---

> 如需 demo 或整合進後續推薦模型，請先確認 Hugging Face LLM 權限已設定正確。
