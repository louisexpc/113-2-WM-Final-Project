# 113-2-WM-Final-Project

## Dataset - H&M Kaggel
- [link of dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
)

### Articles.csv 欄位說

| 欄位名稱 | 中文名稱 | 範例 | 說明 |
|----------|-----------|------|------|
| `article_id` | 商品唯一編號 | `0108775015` | 每一件商品的唯一識別碼 |
| `product_code` | 商品主類別代碼 | `0108775` | 同一款商品的不同變化（如不同顏色）共用一個 `product_code` |
| `prod_name` | 商品名稱 | `Strap top` | 商品的基本命名 |
| `product_type_no` | 商品類型代碼 | `253` | 商品類型的數值編碼 |
| `product_type_name` | 商品類型名稱 | `Vest top` | 商品的類型名稱 |
| `product_group_name` | 商品群組名稱 | `Garment Upper body` | 所屬群組，如上身服飾、下身服飾等 |
| `graphical_appearance_no` | 外觀樣式代碼 | `1010016` | 圖形/樣式的數值編碼 |
| `graphical_appearance_name` | 外觀樣式名稱 | `Solid`, `Stripe` | 外觀樣式名稱，例如素色、條紋等 |
| `colour_group_code` | 顏色群組代碼 | `09` | 顏色編碼 |
| `colour_group_name` | 顏色名稱 | `Black`, `White` | 主觀分類的顏色名稱 |
| `perceived_colour_value_id` | 明暗色系代碼 | `4` | 商品顏色的亮度分類代碼 |
| `perceived_colour_value_name` | 明暗色系名稱 | `Dark`, `Light` | 顏色明暗分類名稱 |
| `perceived_colour_master_id` | 主色系代碼 | `5`, `9` | 感知主色的代碼 |
| `perceived_colour_master_name` | 主色系名稱 | `Black`, `White` | 主觀感知的主色名稱 |
| `department_no` | 部門代碼 | `1676` | 公司內部的部門代碼 |
| `department_name` | 部門名稱 | `Jersey Basic` | 商品所屬的部門名稱 |
| `index_code` | 分類索引代碼 | `A`, `B` | 銷售/分類索引代碼 |
| `index_name` | 分類索引名稱 | `Ladieswear` | 主分類名稱，如女裝 |
| `index_group_no` | 分類群組編號 | `1` | 分類群組代碼 |
| `index_group_name` | 分類群組名稱 | `Ladieswear` | 群組名稱 |
| `section_no` | 商品次分類代碼 | `16` | 更細的分類編號 |
| `section_name` | 商品次分類名稱 | `Womens Everyday Basics` | 商品次分類名稱 |
| `garment_group_no` | 服飾群組代碼 | `1002` | 更細分類的代碼 |
| `garment_group_name` | 服飾群組名稱 | `Jersey Basic`, `Under-, Nightwear` | 商品細分類名稱 |
| `detail_desc` | 商品描述 | `Jersey top with narrow shoulder straps.` | 商品詳細文字描述，可用於 NLP 處理 |

---

📌 **應用建議**：
- 可將 `product_type_name`, `graphical_appearance_name`, `colour_group_name` 等類別轉為 embedding 特徵。
- `detail_desc` 可以透過 NLP 模型轉為語意向量（如 BERT、fastText）。
- `product_code` 可代表同款商品的變體，用於找相似商品。
- 搭配圖片資料（`images/`）做多模態推薦。
