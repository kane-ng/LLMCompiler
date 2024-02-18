import argparse
import asyncio
import os

import dotenv
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler

from configs.dev.configs import CONFIGS as DEV_CONFIGS
from configs.dev.tools import generate_tools
from src.llm_compiler.constants import END_OF_PLAN
from src.llm_compiler.llm_compiler import LLMCompiler
from src.utils.evaluation_utils import arun_and_time
from src.utils.logger_utils import enable_logging

argparser = argparse.ArgumentParser()
argparser.add_argument("--stream", action="store_true", help="stream plan")
argparser.add_argument("--logging", action="store_true", help="logging")
# argparser.add_argument("--store", type=str, required=True, help="store path")
args = argparser.parse_args()

dotenv.load_dotenv()

if args.logging:
    enable_logging(True)
else:
    enable_logging(False)


def get_tools():
    tools = generate_tools(DEV_CONFIGS)
    return tools


def get_configs():
    configs = DEV_CONFIGS
    return configs


async def main():
    configs = get_configs()
    model_name = configs["default_model"]
    tools = get_tools()

    print("Run Octopus")
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    # can be streaming or not
    planner_llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        streaming=args.stream,
    )

    handler = CallbackHandler(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    )
    agent = LLMCompiler(
        tools=tools,
        planner_llm=planner_llm,
        planner_example_prompt=configs["planner_prompt"],
        planner_example_prompt_replan=configs.get("planner_prompt_replan"),
        planner_stop=[END_OF_PLAN],
        planner_stream=args.stream,
        agent_llm=llm,
        joinner_prompt=configs["output_prompt"],
        joinner_prompt_final=configs.get("output_prompt_final"),
        max_replans=configs["max_replans"],
        benchmark=False,
        callbacks=[handler],
    )

    while True:
        question = input("Enter question: ")
        octopus_answer, octopus_time = await arun_and_time(
            agent.arun, question, callbacks=[handler]
        )
        print(f"Answer: {octopus_answer}")
        print("time: ", octopus_time)


if __name__ == "__main__":
    results = asyncio.get_event_loop().run_until_complete(main())
