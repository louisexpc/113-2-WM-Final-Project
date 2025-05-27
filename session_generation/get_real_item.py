import pandas as pd
import pickle
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def find_best_items(input_dict, articles_path='articles.csv', mapping_path='data/article_to_idx.pkl'):
    # load mapping file
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    # load articles
    df = pd.read_csv(articles_path)
    df['full_text'] = df['prod_name'].fillna('') + ' ' + df['detail_desc'].fillna('')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['full_text'])
    user_results = {}
    for user, input_texts in tqdm(input_dict.items()):
        results = []
        print("user", user)
        for input_text in input_texts:
            print("input_text:", input_text)
            input_vec = vectorizer.transform([input_text])
            similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
            top_idx = similarities.argmax()
            best_item = df.iloc[top_idx]
            
            best_item['article_id']
            results.append(
                mapping[best_item['article_id']]
            )
            print("prod_name:", best_item['prod_name'])
            print("detail_desc:", best_item['detail_desc'])
            print("---")
        user_results[user] = results
    return user_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input.pkl 路徑')
    parser.add_argument('--output', type=str, default='results/output.pkl', help='output.pkl 路徑')
    parser.add_argument('--articles', type=str, default='data/articles.csv', help='articles.csv 路徑')
    parser.add_argument('--mapping', type=str, default='data/article_to_idx.pkl', help='article to ids mapping file 路徑')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        input_dict = pickle.load(f)
    user_results = find_best_items(input_dict, articles_path=args.articles)
    print("Total results: ", user_results)
    with open(args.output, 'wb') as f:
        pickle.dump(user_results, f)
