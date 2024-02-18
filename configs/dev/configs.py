from configs.dev.gpt_prompts import OUTPUT_PROMPT, PLANNER_PROMPT

CONFIGS = {
    "model_type": "openai",
    "default_model": "gpt-3.5-turbo-1106",
    "embeddings": "text-embedding-ada-002",
    "prompts": {
        "openai": {
            "planner_prompt": PLANNER_PROMPT,
            "output_prompt": OUTPUT_PROMPT,
        },
    },
    "max_replans": 1,
    "presistent_directory": [
        "resources/chroma_db/llama2_paper_dataset",
        "resources/chroma_db/gemini_paper_dataset",
        "resources/chroma_db/gpt_paper_dataset",
    ],
    "k": 5,
}
