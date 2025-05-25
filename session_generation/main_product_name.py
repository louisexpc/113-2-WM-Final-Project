import os
import pickle
import argparse
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer
from utils.data_loader import load_pickle
from llm2_vllm import get_product_name_from_product_type


def parse_args():
    parser = argparse.ArgumentParser(description="Generate product names from product types using vLLM.")
    parser.add_argument("--gpu", type=int, default=4, help="Which GPU to use (e.g., 0, 1, 5)")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of users to process per batch")
    parser.add_argument("--input_path", type=str, default="data/llama3_enriched_sessions4_final.pkl", help="Path to input session pickle file")
    parser.add_argument("--output_path", type=str, default="data/generated_product_name_4.pkl", help="Path to output result pickle file")
    return parser.parse_args()


def save_result_incrementally(result_dict, output_path):
    """將 result_dict 合併進 pickle 檔案中，若檔案不存在則新建"""
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            existing = pickle.load(f)
    else:
        existing = {}
    existing.update(result_dict)
    with open(output_path, "wb") as f:
        pickle.dump(existing, f)
    print(f"✅ 已儲存 {len(result_dict)} 筆至 {output_path}，目前總筆數：{len(existing)}")
    
    
def load_existing_user_ids(output_path):
    """載入已處理過的 user_id 清單"""
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            existing = pickle.load(f)
        return set(existing.keys())
    return set()


def main():
    args = parse_args()
    
    try:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Set to the GPU you want to use, "4" is 4090
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 改善初始化階段的記憶體碎片問題
        
        # Load the model and tokenizer
        # model_name = "meta-llama/Llama-2-7b-chat-hf"
        model_name = "meta-llama/Llama-3.1-8b-Instruct"
        local_model_dir = "./llama3_hf_cache"
        
        print(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        
        llm = LLM(
            model=local_model_dir,
            gpu_memory_utilization=0.8,  # Adjust based on your GPU memory
            max_model_len=2048,  # Adjust based on your model's max length
            max_num_seqs=16,  # Number of sequences to generate in parallel
        )
        
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=128,
        )
        
        # Example sessions_dict
        # sessions_dict = {
        # '72ec92d25f5ac22812f129ec6e19212a7e1cb1e891748096926e023f2dee1642': [
        #     'Leggings/Tights', 'Leggings/Tights', 'Sweater', 'Leggings/Tights', 'Skirt', 'Leggings/Tights', 
        #     'Sweater', 'Skirt', 'Underwear Tights', 'Underwear set', 'Dress', 'Underwear body', 'Underwear corset', 
        #     'Leg warmers', 'Tailored Waistcoat', 'Swimwear set', 'Sneakers', 'Shorts', 'Trousers', 'Costumes', 
        #     'Outdoor trousers', 'Bodysuit', 'Night gown', 'Hoodie', 'Activewear', 'Sportswear', 'Formalwear', 
        #     'Casualwear', 'Outerwear', 'Accessories'
        #     ],
        # '736b26388527f903df1d4c5099540c6ea54c759828d1c2fdaf125e3992769c3d': [
        #     'Sweater', 'Bra', 'Underwear', 'bottom', 'Underwear', 'bottom', 'Bra', 'Bra', 'Underwear', 'bottom', 
        #     'Underwear bottom', 'Underwear body', 'Underwear corset', 'Swimwear bottom', 'Sweater', 'Bra', 
        #     'Swimwear top', 'Shorts', 'Bikini top', 'Leggings/Tights', 'Garment Set', 'Robe', 'Trousers', 'Skirt', 
        #     'Shirt', 'Night gown', 'Shoes', 'Accessories', 'Dresses', 'Jewelry'
        #     ],
        # "736f7c009447080222e3da0b3dc03a73678202e59d6fcf169c93ab5f939e9319": [
        #     'Underwear', 'Tights', 'Socks', 'Underwear', 'Tights', 'Vest', 'top', 'Vest', 'top', 'T-shirt', 
        #     'Underwear', 'Tights', 'Underwear Tights', 'Leggings/Tights', 'Underwear body', 'Vest top', 
        #     'Kids Underwear top', 'T-shirt', 'Garment Set', 'Swimwear top', 'Tailored Waistcoat', 'Swimwear set', 
        #     'Robe', 'Scarf', 'Shorts', 'Costumes', 'Trousers', 'Swimwear bottom', 'Hoodie', 'Scarf'
        #     ],
        # }   
        
        
        sessions_dict = load_pickle(args.input_path)
        customers_df = pd.read_parquet('data/customers.parquet')
        output_path = args.output_path
        
        # === 載入已處理過的 user_id ===
        processed_users = load_existing_user_ids(output_path)
        print(f"🗂️ 已處理過 {len(processed_users)} 位使用者，將自動略過")

        # === 分批處理每 10 筆 ===
        batch = {}
        count = 0
        skipped = 0
        BATCH_SIZE = args.batch_size  # 每批處理的數量
        for user_id, product_types in sessions_dict.items():
            if user_id in processed_users:
                skipped += 1
                continue  # 🔁 跳過已處理
            batch[user_id] = product_types
            count += 1
            if count % BATCH_SIZE == 0:
                print(f"🚀 處理第 {count - 9 + skipped} 到 {count + skipped} 筆")
                result = get_product_name_from_product_type(
                    product_type_dict=batch,
                    customers_df=customers_df,
                    llm=llm,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params
                )
                save_result_incrementally(result, output_path)
                batch = {}  # 清空批次
                
        # 處理剩餘不足 10 筆的資料
        if batch:
            print(f"🚀 處理剩餘最後 {len(batch)} 筆")
            result = get_product_name_from_product_type(
                product_type_dict=batch,
                customers_df=customers_df,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params
            )
            save_result_incrementally(result, output_path)

        
        # result = get_product_name_from_product_type(
        #     product_type_dict=sessions_dict, 
        #     customers_df=customers_df, 
        #     llm=llm, 
        #     tokenizer=tokenizer,
        #     sampling_params=sampling_params
        # )
        
        # print("Result:", result)

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()