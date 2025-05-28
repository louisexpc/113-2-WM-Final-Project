import os
import pickle 
import pandas as pd

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


def customers_id_mapping(old_data, mapping_dict):
    """
    Map old customer IDs to new ones based on a mapping dictionary.
    
    Args:
        old_data (pd.DataFrame): DataFrame containing old customer IDs
        mapping_dict (dict): Dictionary mapping old IDs to new IDs
        
    Returns:
        pd.DataFrame: DataFrame with updated customer IDs
    """
    if not isinstance(old_data, dict):
        print("Error: old_data should be a dictionary")
        return None
    
    # 創建新的字典，將舊的客戶ID映射到新的ID
    mapping_data = {}
    for old_id, value in old_data.items():
        # 如果在映射字典中找到對應的新ID，使用新ID；否則保持原ID
        new_id = mapping_dict.get(old_id, old_id)
        mapping_data[new_id] = value
    return mapping_data


def main():
    old_data = load_pickle(os.path.join("data","real_item_6.pkl"))
    if old_data is None:
        return
    # old_data = pd.DataFrame.from_dict(old_data, orient='index').reset_index()
    # print(old_data.head())
    # print(type(old_data))
    mapping_dict = load_pickle(os.path.join("data","customer_to_idx.pkl"))
    # print(type(mapping_dict))
    mapping_data = customers_id_mapping(old_data, mapping_dict)
    print(mapping_data[1383])
    
    original_dataset = load_pickle(os.path.join("data","sessions_5_4_30_mapping.pkl"))
    original_dataset = pd.DataFrame.from_dict(original_dataset, orient='index').reset_index()
    original_dataset = original_dataset.iloc[:, :2]  # 選擇前兩列
    original_dataset.columns = ['index', 'article_id']
    # Add a new column that stores the length of each list in the article_id column
    original_dataset['K'] = original_dataset['article_id'].apply(len)
    print("Dataset with list length column:")
    print(original_dataset.head())
    new_data = {}
    for index, session in mapping_data.items():
        matching_rows = original_dataset[original_dataset['index'] == index]
        if not matching_rows.empty:
            # 幾種取單一值的方式（都是等價的）:
            top_k = matching_rows['K'].item()          # 使用 iloc[0]
            top_k -= 1
            article_ids = matching_rows['article_id'].iloc[0]
            
            # 直接創建新的列表
            new_session = article_ids[:top_k] + session[top_k:] + article_ids[top_k:]
            new_data[index] = new_session
            
    # Save the updated data to a new pickle file
    output_path = os.path.join("data", "real_item_6_fix.pkl")
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(new_data, f)
        print(f"Updated data saved to {output_path}")
        
        
        for i in range(5):
            id = list(mapping_data.keys())[i]
            print(f"Session {id}: {mapping_data[id]}")
            print("New session:")
            print(f"Session {id}: {new_data[id]}")
            print("original session:")
            print(f"Session {id}: {original_dataset[original_dataset['index'] == id]['article_id'].values[0]}")

    except Exception as e:
        print(f"Error saving data to {output_path}: {str(e)}")
        
        
    
    """
    """
            
    

if __name__ == "__main__":
    main()