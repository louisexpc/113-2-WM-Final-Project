# LLM1 product_type generation

Environments
---
python 3.10.16
```bash
pip install -r requirements.txt
```

Data Path
---
sessions_5_4.pkl & sessions_5_6.pkl in [here](https://drive.google.com/drive/folders/15yY3Y58dTSp_yLDWK5TkhHQZEE-My069)
```
llm1
├── dataset/
│ ├── get_product_type_from_history_prompt.txt
│ ├── product_types.csv
│ ├── sessions_5_4.pkl
│ ├── sessions_5_6.pkl
│ └── article_to_product_mapping.pkl
├── module.py
├── utils.py
└── main.py
```

Model
---

```
meta-llama/Llama-2-7b-chat-hf
```
Parameters
---
```
# line 7 in main,py 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

--model_name: HuggingFace model name
--session_path: Path to user session .pkl file
--category_csv: Path to product_types.csv
--prompt_path: Path to LLM prompt .txt file
--save_path: Checkpoint save path
--final_save_path: Final output save path
--batch_size: Save every N users
--max_users: Maximum number of users to process
--total_len: Total session length after enrichment
--top_k: Number of top categories for similarity filtering
```

Run the following command
---
```
python main.py
or 
bash run.sh
```
