"""
preview.py
這份程式用以確認 pickle 檔案的內容。
主要可以確認生成的 product name 是否有包含原始 article_id

"""
import pickle
import pprint

def preview_pickle_file(file_path):
    """
    Load and display the contents of a pickle file.
    
    Args:
        file_path (str): Path to the pickle file
    """
    
        # Open and load the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Print basic information about the data
    print(f"Data type: {type(data)}")
    print(f"Total items: {len(data)}")
    print(data[8265]) # Display first item if it's a dictionary
        
        
        
        
def preview_ori_pickle_file(file_path):
    """
    Load and display the contents of a pickle file.
    
    Args:
        file_path (str): Path to the pickle file
    """
    try:
        # Open and load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Print basic information about the data
        print(f"Data type: {type(data)}")
        
        # Handle different data types appropriately
        print(data[8265]["article_id"])
        
        # Print the actual data
        print("\nData content:")
        # pprint.pprint(data, depth=4, compact=False)
        
    except Exception as e:
        print(f"Error reading pickle file: {e}")

if __name__ == "__main__":
    file_path = "dataset/generated_real_name_6_test.pkl"  # Replace with your file path
    preview_pickle_file(file_path)
    ori_file_path = "dataset/short_sessions_5_6_30_mapping.pkl"  # Replace with your file path
    preview_ori_pickle_file(ori_file_path)