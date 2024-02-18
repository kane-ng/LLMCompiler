from src.agents.tools import Tool
from src.tools.knowledge_tool.react_retrieval import ReActRetrieval


def generate_tools(cfg):
    tools = []
    for persist_directory in cfg["presistent_directory"]:
        retrieval_chain = ReActRetrieval.setup_retrieval_system(
            llm_model=cfg["default_model"],
            embeddings_model=cfg["embeddings"],
            persist_directory=persist_directory,
            chain_type="stuff",
            k=cfg["k"],
        )

        tool_name = f"{persist_directory.split('/')[-1]}_search".lower()
        tool = Tool(
            name=tool_name,
            func=retrieval_chain.ainvoke,
            return_direct=True,
            description=(
                f"{tool_name}({{'query': query}}) -> str:\n"
                " - Executes a search for the query on vectorstore.\n"
                " - Returns the results of the RAG serach and the source documents.\n"
                " - `query`: query to search for on vector database, e.g., What are the different kinds of pets people commonly own?."
            ),
            stringify_rule=lambda args: f"{tool_name}({args[0]})",
        )
        tools.append(tool)

    return tools
