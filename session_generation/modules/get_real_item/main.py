import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json

def build_sparse_matrix(texts, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        return vectorizer, matrix
    elif method == 'bm25':
        # BM25 的語料要斷詞
        tokenized = [str(t).split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        return bm25, tokenized
    else:
        raise ValueError("method must be 'tfidf' or 'bm25'")

def find_best_items(cfg, logger):
    """Origin Parameter
     articles_path, mapping_path, method="bm25", dense_model_name="all-MiniLM-L6-v2", hybrid_alpha=0.5, log_path=None
    """
    logger.info(f"🔧 使用檢索模式：{method}")

    method = cfg.retrieval.method
    hybrid_alpha = cfg.retrieval.hybrid_alpha
    dense_model_name = cfg.retrieval.dense_model
    if torch.cuda.is_available() and cfg.device.cuda_visible_devices:
        device = f"cuda:{cfg.device.cuda_visible_devices}"
    else:
        device = "cpu"
   

    logger.info(f"📂 讀取 input_dict 檔案：{cfg.data.input_path}")
    try:
        with open(cfg.data.input_path, 'rb') as f:
            input_dict = pickle.load(f)
    except Exception as e:
        raise ValueError(f'Loading input dict {cfg.data.input_path} failed: {e}')

    
    logger.info(f"📂 讀取 mapping 檔案：{cfg.data.mapping_path}")

    with open(cfg.data.mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    
    logger.info(f"📂 讀取商品資料：{cfg.data.articles_csv}")

    df = pd.read_csv(cfg.data.articles_csv)
    df['full_text'] = df['prod_name'].fillna('') + ' ' + df['detail_desc'].fillna('')
    # 預處理
    article_texts = df['full_text'].astype(str).tolist()
    

    # 建 sparse/bm25 索引
    if method == "tfidf":
        logger.info("🔍 建立 TF-IDF 向量")
        sparse_vectorizer, sparse_matrix = build_sparse_matrix(article_texts, 'tfidf')
    elif method == "bm25":
        logger.info("🔍 建立 BM25 索引")
        bm25, bm25_tokenized = build_sparse_matrix(article_texts, 'bm25')
    elif method in ("hybrid", "dense"):
        # 需要 dense model
        logger.info(f"⚙️ 載入 dense model: {dense_model_name}")
        from sentence_transformers import SentenceTransformer
        dense_model = SentenceTransformer(dense_model_name, device=device)
        dense_embs = dense_model.encode(article_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        if method == "hybrid":
            logger.info("🔍 混合模式啟用：建立 BM25 索引")
            bm25, bm25_tokenized = build_sparse_matrix(article_texts, 'bm25')
    else:
        raise ValueError("Unknown method.")

    # 開一個文字檔用來紀錄 (可選擇 log_path 參數)


    user_results = {}
    logger.info(f"👤 總共使用者數：{len(input_dict)}")

    for user, input_texts in tqdm(input_dict.items(), desc='users'):
        results = []
        for input_text in input_texts:
            print("input_text:", input_text)
            # 1. Sparse retrieval
            if method == "tfidf":
                input_vec = sparse_vectorizer.transform([input_text])
                scores = (input_vec * sparse_matrix.T).toarray()[0]
                top_idx = scores.argmax()
            elif method == "bm25":
                input_tokenized = input_text.split()
                scores = bm25.get_scores(input_tokenized)
                top_idx = int(np.argmax(scores))
            elif method == "dense":
                input_emb = dense_model.encode([input_text], normalize_embeddings=True)
                scores = np.dot(dense_embs, input_emb[0])
                top_idx = int(np.argmax(scores))
            elif method == "hybrid":
                # Dense
                input_emb = dense_model.encode([input_text], normalize_embeddings=True)
                dense_scores = np.dot(dense_embs, input_emb[0])
                # BM25
                input_tokenized = input_text.split()
                sparse_scores = bm25.get_scores(input_tokenized)
                # score normalize
                scaler = MinMaxScaler()
                dense_scores_norm = scaler.fit_transform(dense_scores.reshape(-1, 1)).flatten()
                sparse_scores_norm = scaler.fit_transform(np.array(sparse_scores).reshape(-1, 1)).flatten()
                # Hybrid 融合
                scores = hybrid_alpha * dense_scores_norm + (1 - hybrid_alpha) * sparse_scores_norm
                top_idx = int(np.argmax(scores))
            else:
                raise ValueError("Unknown method.")

            best_item = df.iloc[top_idx]
            results.append(mapping[best_item['article_id']])
            if cfg.output.enable_detailed_log:
                logger.info(f"[User {user}] input: {input_text}")
                logger.info(f"[User {user}] → prod_name: {best_item['prod_name']}")
                logger.info(f"[User {user}] → detail_desc: {best_item['detail_desc']}")

            
        
        user_results[user] = results
        logger.info("✅ 檢索完成...儲存中")

        with open(cfg.output.result_path, 'wb') as f:
            pickle.dump(user_results, f)
        
        logger.info(f"📦 結果已儲存到 {cfg.output.result_path}")

    return user_results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', type=str, required=True, help='input.pkl 路徑')
#     parser.add_argument('--output', type=str, default='results/output.pkl', help='output.pkl 路徑')
#     parser.add_argument('--articles', type=str, default='data/articles.csv', help='articles.csv 路徑')
#     parser.add_argument('--mapping', type=str, default='data/article_to_idx.pkl', help='article to ids mapping file 路徑')
#     parser.add_argument('--method', type=str, default='bm25', choices=['bm25', 'tfidf', 'dense', 'hybrid'], help='檢索方式')
#     parser.add_argument('--dense_model', type=str, default='all-MiniLM-L6-v2', help='dense retrieval 模型名稱')
#     parser.add_argument('--hybrid_alpha', type=float, default=0.5, help='hybrid 模式下 dense 與 sparse 分數權重 (0~1)')
#     parser.add_argument('--log', type=str, default='results/output.json', help='Optional: 紀錄 input+output 到的 txt 路徑')

#     args = parser.parse_args()

#     with open(args.input, 'rb') as f:
#         input_dict = pickle.load(f)
#     user_results = find_best_items(
#         input_dict, 
#         articles_path=args.articles, 
#         mapping_path=args.mapping,
#         method=args.method,
#         dense_model_name=args.dense_model,
#         hybrid_alpha=args.hybrid_alpha,
#         log_path=args.log
#     )
#     with open(args.output, 'wb') as f:
#         pickle.dump(user_results, f)