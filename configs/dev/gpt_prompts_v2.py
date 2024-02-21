from src.llm_compiler.constants import END_OF_PLAN, JOINNER_FINISH

PLANNER_PROMPT = (
    "For example, you will be given two collections of vector and a question. The collections can be related to each other or not.\n"
    " The tool you are provided with will have the name following the pattern: {{collection_name}}_search: llama2_paper_dataset_search and eval_llm_survey_paper_dataset.\n"
    " You will need to search for the answer to the question in each of the collection or search in all the collections and then join the results to answer the question.\n"
    "\n"
    'Question: Based on the abstract of "Llama 2: Open Foundation and Fine-Tuned Chat Models," what are the two primary objectives achieved in this work, and what is the range of parameters for the large language models developed?\n'
    '1. llama2_paper_dataset_search("What are the two primary objectives achieved in Llama 2?")\n'
    '2. llama2_paper_dataset_search("What is the range of parameters for the large language models developed?")\n'
    "Thought: I can answer the question now.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: What is the purpose of ToolQA and how does it differ from other benchmarks mentioned in the document?\n"
    '1. eval_llm_survey_paper_dataset("What is the purpose of ToolQA?")\n'
    '2. eval_llm_survey_paper_dataset("How does ToolQA differ from other benchmarks mentioned in the document?")\n'
    "Thought: I can answer the question now.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: What is Commonsense Reasoning and which benchmarks are used for assessing Commonsense Reasoning in the evaluation of Llama 2 pretrained models?\n"
    '1. eval_llm_survey_paper_dataset("What is Commonsense Reasoning?")\n'
    '2. llama2_paper_dataset_search("Which benchmarks are used for assessing Commonsense Reasoning in the evaluation of Llama 2 pretrained models?")\n'
    "Thought: I can answer the question now.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
)

OUTPUT_PROMPT = (
    "Solve a question answering task with interleaving Observation, Thought, and Action steps. Here are some guidelines:\n"
    "  - You will be given a Question and some Wikipedia passages, which are the Observations.\n"
    "  - Thought needs to reason about the question based on the Observations in 1-2 sentences.\n"
    "  - There are cases where the Observations are unclear or irrelevant (in the case wikipedia search was not successful). In such a case where the Observations are unclear, you must make a best guess based on your own knowledge if you don't know the answer. You MUST NEVER say in your thought that you don't know the answer.\n\n"
    "Action can be only one type:\n"
    f" (1) {JOINNER_FINISH}(answer): returns the explaination of the answer base on thought and finishes the task. "
    "Answer should not be too short as it must contain the explaination and MUST not be multiple choices. Answer MUST NEVER be 'unclear', 'unknown', 'neither', 'unrelated' or 'undetermined', and otherwise you will be PENALIZED.\n"
    "\n"
    "Here are some examples:\n"
    "\n"
    "Question: Which magazine was started first Arthur's Magazine or First for Women?\n"
    "\n"
    "search(Arthur's Magazine)\n"
    "Observation: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.\n"
    "search(First for Women (magazine))\n"
    "Observation: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.\n"
    "Thought: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.\n"
    f"Action: {JOINNER_FINISH}(First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.)\n"
    "###\n"
    "\n"
    "Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n"
    "search(Pavel Urysohn)\n"
    "Observation: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\n"
    "search(Leonid Levin)\n"
    "Observation: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.\n"
    "Thought: Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.\n"
    f"Action: {JOINNER_FINISH}(Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.)\n"
    "###\n"
    "\n"
)
