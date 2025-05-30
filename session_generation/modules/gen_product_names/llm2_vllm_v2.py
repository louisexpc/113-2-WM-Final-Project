import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizer
from .utils.core_helpers import get_product_name_examples, extract_product_name, get_user_info

# 呼叫這個 function 前，要先將 customers.parquet 讀進 pd.DataFrame

def get_product_name_from_product_type_vllm(
    product_type_dict: dict[str, tuple[list[str], str]], 
    customers_df: pd.DataFrame, 
    llm=LLM, 
    tokenizer=PreTrainedTokenizer,
    sampling_params=SamplingParams
) -> dict[str, list[str]]:
    """為每個使用者的每個類別生成對應的商品名稱
    
    Args:
        product_type_dict: {user_id: ([product_type1, product_type2, ...], k)}
        customers_df: pd.DataFrame, 使用者資料，包含 age, fashion_news_frequency, club_member_status

    Returns:
        dict[str, list[str]]: {user_id: [product_name1, product_name2, ...]}
        每個 product_name 對應 product_type_dict 中相同位置的 product_type
    """
    result = {}

    for user_id, (product_types, k) in tqdm(product_type_dict.items(), desc="Processing users"):
        user_info = get_user_info(user_id, customers_df)
        if user_info.empty:
            print(f"⚠️ No user info found for {user_id}")
            result[user_id] = ["Unknown"] * len(product_types)
            continue
        prompts = []
        user_map = []
        for i, product_type in enumerate(product_types):
            if i < k-1:
                continue
            
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
            7. Product names should be enclosed in square brackets, e.g., [Product Name].

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

            Output Format: [Product Name]
            Output Example: [Knit dress]
            """
        
            # 生成回應
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
            user_map.append((user_id, i))
            
        outputs = llm.generate(prompts, sampling_params)
        print(f"Generated {len(outputs)} outputs for {len(user_map)} prompts.")
        
        
        for (user_id, index), output in zip(user_map, outputs):
            raw_text = output.outputs[0].text.strip()
            product_name = extract_product_name(raw_text)
            result[user_id][index] = product_name
    
    return result