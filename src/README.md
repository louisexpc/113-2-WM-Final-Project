# LLM-based User Session Processing Pipeline
This project is a multi-stage pipeline designed to process user Browse sessions, enrich them with generated product categories and names using Large Language Models (LLMs), and finally map these generated names to real products in an inventory.

The core workflow consists of three main modules:

1. Enrich Session: Takes a user's session history (a sequence of product IDs) and uses an LLM to predict a relevant product category that the user might be interested in next. It then creates a new, "enriched" session.
2. Generate Product Names: Takes the enriched session (with the added product category) and uses an LLM to generate a hypothetical, specific product name that fits the predicted category and user history.
3. Get Real Item: Takes the generated product name and uses a retrieval system (like TF-IDF or BM25) to find and map it to the most similar real item from a product database.

The final output is a pickle file containing user sessions where the last item is a recommended product ID derived from this entire process.

## Directory Structure
```
.
├── .env                  # Environment variables, e.g., Hugging Face token
├── README.md             # This README file
├── requirements.txt      # Python dependencies
├── run.py                # Main Python entry point for the pipeline
├── run.sh                # Master script to execute the entire pipeline
└── src
    ├── config
    │   ├── enrich_sessions.yaml      # Config for the session enrichment module
    │   ├── gen_product_names.yaml    # Config for the product name generation module
    │   └── get_real_item.yaml        # Config for the real item mapping module
    ├── dataset
    │   ├── articles.csv              # Product catalog data
    │   ├── customers_mapping.csv     # Customer data
    │   ├── download.sh               # Script to download necessary data files
    │   └── ... (other .pkl and .csv files for I/O)
    ├── modules
    │   ├── enrich_session
    │   │   ├── main.py               # Logic for LLM 1 (session enrichment)
    │   │   └── ...
    │   ├── gen_product_names
    │   │   ├── main.py               # Logic for LLM 2 (product name generation)
    │   │   └── ...
    │   └── get_real_item
    │       └── main.py               # Logic for retrieval (TF-IDF, BM25, etc.)
    ├── prompts
    │   └── get_product_type_from_history_prompt_v1.1.txt # Prompt template for LLM 1
    └── utils
        ├── config.py                 # Utilities for loading YAML configs
        ├── logger.py                 # Setup for logging
        └── ...
```

## Execution Instructions

1. Prerequisites
    - Python: Ensure you have a compatible Python version installed (e.g., Python 3.10+).
    - Dependencies: Install the required Python packages using `requirements.txt`.
        ```Bash
        pip install -r requirements.txt
        ```
    - Environment Variables: The project uses a `.env` file to manage sensitive information. Create a `.env` file in the project root and add your Hugging Face token:
        ```
        HF_TOKEN="your_hugging_face_api_token"
        ```
        This is required by the `enrich_session` module to download and use the specified LLM from the Hugging Face Hub.
2. Running the Pipeline
The easiest way to run the entire pipeline is by using the provided shell script. It handles data downloads and executes each step in the correct order.
    ```
    # Make the script executable
    chmod +x run.sh

    # Run the entire pipeline
    ./run.sh
    ```
    The `run.sh` script performs the following actions:
    1. Downloads Data: Navigates to `src/dataset` and runs download.sh.
    2. Executes Main Pipeline: Runs `python run.py`, which executes the three main modules sequentially.
    3. Post-processing: Runs a final `generate_testset_from_sessions.py` script on the output.

3. Alternative: Manual Execution
You can also run the main Python script directly.
    ```
    python run.py
    ```
    `run.py` also accepts command-line arguments. For instance, to only run the final item mapping module (useful if you already have the generated product names), use the `--mapping` flag:
    ```
    python run.py --mapping
    ```

## Configuration (YAML Files)

The behavior of each module is controlled by a corresponding YAML file in `src/config/`. The data paths in these files are chained: the output of one module becomes the input for the next.

1. `config/enrich_sessions.yaml`
This file configures the first LLM task, which enriches user sessions with a predicted product category.
    - `model`:
        - `name`: The Hugging Face model identifier for the LLM (e.g., `meta-llama/Llama-3.1-8b-Instruct`).
        - `top_k`: The number of candidate categories to pre-filter using sentence embeddings before passing them to the LLM.
        - `total_len`: The desired total length of the user session after enrichment.
        - `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`: vLLM parameters for model inference.
    - `short_model`:
        - `name`: The SentenceTransformer model used for initial category filtering (e.g., `all-MiniLM-L6-v2`).
    - `data`:
        - `session_path`: Input file with the raw user sessions.
        - `category_csv`: Path to the CSV file containing all possible product categories.
        - `prompt_path`: The prompt template file for the LLM.
    - `save`:
        - `final_output_path`: Output file where the enriched sessions are saved. This path is used as the input for the next module.
        - `batch_size`, `max_users`: Parameters for checkpointing and limiting the number of processed users.
    - `hf`:
        - `use_token`: Set to true to login to Hugging Face.
        - `token_env_key`: The environment variable key for the HF token (e.g., `HF_TOKEN`).


2. `config/gen_product_names.yaml`  
This file configures the second LLM task, which generates a product name based on the enriched session.
- `model`:
    - `name`: The same LLM as the first module. The script is optimized to pass the loaded model directly, so this is mainly for reference.
    - `temperature`, `top_p`, `max_tokens`: Sampling parameters for the LLM to control the creativity and length of the generated names.
- `data`:
    - `input_path`: Input file. Must match `save.final_output_path` from `enrich_sessions.yaml`.
    - `customers_path`: Path to customer data, potentially used to personalize the generated name.
    - `output_path`: Output file where the generated product names are saved.
    - `product_examples_path`: Path to a file with few-shot examples of product names to improve generation quality.
    - `batch_size`: The number of users to process in a single batch.

3. `config/get_real_item.yaml`  
This file configures the final module, which maps the generated product name to a real product.  
    - `retrieval`:
        - `method`: The core algorithm for finding the best item. Options:
            - `tfidf`: Uses TF-IDF vector similarity.
            - `bm25`: Uses the BM25 ranking function.
            - `dense`: Uses a SentenceTransformer model for dense vector similarity.
            - `hybrid`: A weighted combination of dense and bm25 scores.
        - `hybrid_alpha`: The weighting factor for the hybrid method. A value of 1.0 is fully dense, and 0.0 is fully bm25.
        - `dense_model`: The SentenceTransformer model to use for the `dense` or `hybrid` methods.
    - `data`:
        - `input_path`: Input file. Must match data.output_path from gen_product_names.yaml.
        - `articles_csv`: Required. The main product catalog containing `prod_name` and `detail_desc` for all real items.
        - `mapping_path`: Required. A pickle file mapping `article_id` to an internal index.
    - `output`:
        - `result_path`: Final Output of the entire pipeline. This file contains the final processed sessions with the mapped real item ID.
    - `device`:
        - `cuda_visible_devices`: Specifies the GPU to use, which is highly recommended for `dense` or `hybrid` retrieval methods.