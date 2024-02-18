from configs.dev.gpt_prompts import OUTPUT_PROMPT, PLANNER_PROMPT

CONFIGS = {
    "default_model": "gpt-3.5-turbo-1106",
    "embeddings": "text-embedding-ada-002",
    "planner_prompt": PLANNER_PROMPT,
    "output_prompt": OUTPUT_PROMPT,
    "max_replans": 0,
    "presistent_directory": [
        "resources/chroma_db/BlockchainSolanaDataset",
        "resources/chroma_db/Llama2PaperDataset",
    ],
    "k": 5,
}
