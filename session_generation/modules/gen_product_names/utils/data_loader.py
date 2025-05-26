import re
import os
import ast
import tqdm
import pickle
import pandas as pd
from datetime import datetime


def load_session_file(file_path) -> list[dict]:
    """python
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


def load_pickle(pickle_path):
    """
    Load user-item data from a pickle file.
    
    Args:
        pickle_path (str): Path to the pickle file
        
    Returns:
        dict: Dictionary with user IDs as keys and lists of item IDs as values
        Format: {uid_1: [iid_0, iid_1, iid_2, ...]}
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {pickle_path} not found")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {str(e)}")
        return None
    
    
def load_parquet(parquet_path):
    """
    Load a parquet file into a pandas DataFrame.
    
    Parameters:
    parquet_path (str): The path to the parquet file.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        # Check if the file exists
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"File not found: {parquet_path}")
        
        # Load the parquet file
        df = pd.read_parquet(parquet_path)
        
        # Print basic info for verification
        print(f"Loaded parquet file: {parquet_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while loading the parquet file: {e}")
        
        