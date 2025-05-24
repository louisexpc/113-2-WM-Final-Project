# LLM1 product_type generation

Environments
---
python 3.10.16
```bash
pip install -r requirements.txt
```

Data Path
---

```
llm1
├── dataset/
│ ├── get_product_type_from_history_prompt.txt
│ ├── article_to_product_mapping.pkl
│ ├── product_types.csv
│ ├── sessions_5_4.pkl
│ ├── sessions_5_6.pkl
│ └── article_to_product_mapping.pkl
├── module.py
├── utils.py
└── main.py
```

Model
```
meta-llama/Llama-2-7b-chat-hf
```
Run the following command
---
```
python main.py
or 
bash run.sh
```
