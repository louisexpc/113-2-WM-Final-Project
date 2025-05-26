import pickle
import pandas as pd

def load_pickle(path):
    try:
        with open(path, 'rb') as file:
            file = pickle.load(file)
        return file
    except Exception as e:
        print(f"Loading pickle {path} faild: {e}")
    return None
def save_pickle(file, path):
    try:
        with open(path,'wb') as f:
            pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"save pickle {path} faild: {e}")

def articleID_to_product_type_name(article_df:pd.DataFrame, article_ids:list)->list[str]:
    """
    Parameters:
    - article_df: pd.Dataframe, must contain with cols ['article_id','product_type_name'].
    - article_ids: list of int, contain article ids of the user.

    Return:
    - category: list of string.
    """
    article_df['article_id'] = article_df['article_id'].astype(int)
    article_df['product_type_name'] = article_df['product_type_name'].astype(str)
    categroy = article_df[article_df['article_id'].isin(article_ids)]['product_type_name']
    return categroy.to_list()

def load_prompt(file_path: str, N: int) -> str:
    """
    Load a prompt template from a file and replace `{N}` with the given integer.

    Args:
        file_path (str): The path to the prompt.txt file.
        N (int): The integer value to replace `{N}` with.

    Returns:
        str: The formatted prompt string with `{N}` replaced.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt.replace('{N}', str(N))
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the prompt: {e}")

if __name__ == '__main__':
    # path = "pseudo_session.pkl"
    # article_df = pd.read_csv(
    #     'articles_mapping.csv'
    # )
    # # psedu_sessions = dict()
    # data = load_pickle(path)
    # for id,d in data.items():
    #     category = articleID_to_product_type_name(article_df,d['article_id'])
    #     print(f"{id} : {category}")
    # top_k = 5
    # count = 0
    # for key, val in data.items():
    #     psedu_sessions[key] = val

    #     if(count>top_k): break
    #     else : count+=1
    # save_pickle(psedu_sessions,"pseudo_session.pkl")
    # print(psedu_sessions)
    prompt_file = "prompt_v1.txt"
    print(load_prompt(prompt_file,5))