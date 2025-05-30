import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
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

def find_best_items(input_dict, articles_path, mapping_path, method="bm25", dense_model_name="all-MiniLM-L6-v2", hybrid_alpha=0.5, log_path=None):
    # 讀入 mapping, articles
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    df = pd.read_csv(articles_path)
    df['full_text'] = df['prod_name'].fillna('') + ' ' + df['detail_desc'].fillna('')

    # 預處理
    article_texts = df['full_text'].astype(str).tolist()

    # 建 sparse/bm25 索引
    if method == "tfidf":
        sparse_vectorizer, sparse_matrix = build_sparse_matrix(article_texts, 'tfidf')
    elif method == "bm25":
        bm25, bm25_tokenized = build_sparse_matrix(article_texts, 'bm25')
    elif method in ("hybrid", "dense"):
        # 需要 dense model
        dense_model = SentenceTransformer(dense_model_name, device='cuda')
        dense_embs = dense_model.encode(article_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        if method == "hybrid":
            bm25, bm25_tokenized = build_sparse_matrix(article_texts, 'bm25')
    else:
        raise ValueError("Unknown method.")

    # 開一個文字檔用來紀錄 (可選擇 log_path 參數)
    log_f = open(log_path, 'w', encoding='utf-8') if log_path else None

    user_results = {}

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
            print("mapping prod_name:", best_item['prod_name'])
            print("mapping detail_desc:", best_item['detail_desc'])
            print("---")
            # 這行就是寫入 log
            if log_f:
                json.dump({
                    "user_id": user,
                    "input_text": input_text,
                    "mapping_prod_name": best_item['prod_name'],
                    "mapping_detail_desc": best_item['detail_desc']
                }, log_f, ensure_ascii=False)
                log_f.write('\n')
        
        user_results[user] = results
    return user_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input.pkl 路徑')
    parser.add_argument('--output', type=str, default='results/output.pkl', help='output.pkl 路徑')
    parser.add_argument('--articles', type=str, default='data/articles.csv', help='articles.csv 路徑')
    parser.add_argument('--mapping', type=str, default='data/article_to_idx.pkl', help='article to ids mapping file 路徑')
    parser.add_argument('--method', type=str, default='bm25', choices=['bm25', 'tfidf', 'dense', 'hybrid'], help='檢索方式')
    parser.add_argument('--dense_model', type=str, default='all-MiniLM-L6-v2', help='dense retrieval 模型名稱')
    parser.add_argument('--hybrid_alpha', type=float, default=0.5, help='hybrid 模式下 dense 與 sparse 分數權重 (0~1)')
    parser.add_argument('--log', type=str, default='results/output.json', help='Optional: 紀錄 input+output 到的 txt 路徑')

    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        input_dict = pickle.load(f)
    user_results = find_best_items(
        input_dict, 
        articles_path=args.articles, 
        mapping_path=args.mapping,
        method=args.method,
        dense_model_name=args.dense_model,
        hybrid_alpha=args.hybrid_alpha,
        log_path=args.log
    )
    with open(args.output, 'wb') as f:
        pickle.dump(user_results, f)
