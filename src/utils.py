import pandas as pd
import os.path as path
import re
import ast
from tqdm import tqdm

def load_session_file(file_path) -> list[dict]:
    """
    Read a parquet file at path and parse the sessions column into a list of dicts:

    Returns:
        List[dict]: [
            {
                "customer_id": int,
                "sequences": {
                    "articles_id": List[int],
                    "prices": List[float],
                    "timestamp": List[pd.Timestamp],
                    "channels": List[int]
                }
            },
            ...
        ]
    """
    df = pd.read_parquet(file_path, engine='pyarrow')

    def parse_sessions(sessions_str):
        
        sessions_str_clean = re.sub(r"Timestamp\('([^']+)'\)", r"'\1'", sessions_str)
        parsed_tuple = ast.literal_eval(sessions_str_clean)
        articles_id = parsed_tuple[0]
        prices = parsed_tuple[1]
        timestamps_str = parsed_tuple[2]
        channels = parsed_tuple[3]
        timestamps = pd.to_datetime(timestamps_str)
        return {
            'articles_id': articles_id,
            'prices': prices,
            'timestamp': list(timestamps),
            'channels': channels
        }

    result = []
    for _, row in tqdm(df.iterrows(),unit=" row",desc="Loading"):
        result.append({
            'customer_id': row['customer_id'],
            'sequences': parse_sessions(row['session'])
        })
        
    return result

    

def main():
    file = path.join("eda","session_10_filtered_10.parquet")
    df = load_session_file(file)
    raw = df[0]
  
        
    for k,v in raw['sequences'].items():
        print(f"{k}:\n")
        for e in v:
            print(f"({e}, {type(e)})")
    

if __name__ == "__main__":
    main()