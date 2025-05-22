import pandas as pd
from utils.core_helpers import get_current_season, get_product_name_examples, extract_product_name
# 呼叫這個 function 前，要先將 customers.parquet 讀進 pd.DataFrame

def get_product_name_from_product_type(product_type_dict: dict[str, list[str]], user_info: pd.DataFrame, model, tokenizer) -> dict[str, list[str]]:
    """為每個使用者的每個類別生成對應的商品名稱
    
    Args:
        product_type_dict: {user_id: [product_type1, product_type2, ...]}
        user_info: pd.DataFrame, 使用者資料，包含 age, fashion_news_frequency, club_member_status
        
    Returns:
        dict[str, list[str]]: {user_id: [product_name1, product_name2, ...]}
        每個 product_name 對應 product_type_dict 中相同位置的 product_type
    """
    current_season = get_current_season()
    result = {}
    
    for user_id, product_types in product_type_dict.items():
        if user_info.empty:
            print(f"⚠️ No user info found for {user_id}")
            result[user_id] = ["Unknown"] * len(product_types)
            continue
        
        product_names = []
        
        for product_type in product_types:
            
            # 為這個使用者的所有類別生成商品名稱
            system_prompt = """
            You are a fashion product naming assistant. 
            You MUST:
            1. Output ONLY ONE product name
            2. Follow the format strictly
            3. Never include explanations
            4. Never include quotes or brackets
            5. Never include emojis
            6. Never include any other text like "Sure, here is a product name for a jacket that meets the requirements:"

            If you fail to follow these rules, the output will be rejected."""
            user_prompt = f"""
            Generate **ONE** product name for {product_type} that is:
            1. 2-4 words long
            2. Unique and creative

            Customer context:
            - Age: {user_info["age"]}
            - Fashion magazine subscription: {user_info["fashion_news_frequency"]}
            - Club status: {user_info["club_member_status"]}

            Example: {get_product_name_examples(product_type)}

            Output Format: Product Name
            Output Example: Knit dress
            """
        
            # 生成回應
            full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt.strip()} [/INST]"
            inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)  # 增加 token 數以容納多個名稱
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            response = response.split("[/INST]")[-1].strip()
            print(f"Response for {user_id} - {product_type}:")
            print(response)
            print("===" * 20)
        
        # 解析回應
            product_name = extract_product_name(response, fallback=f"Unknown {product_type}")
            product_names.append(product_name)
        result[user_id] = product_names
        
    return result