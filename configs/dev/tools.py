from src.agents.tools import Tool
from src.tools.knowledge_tool.react_retrieval import ReActRetrieval


def generate_tools(cfg):
    retrieval_chain = ReActRetrieval.setup_retrieval_system(
        llm_model=cfg["default_model"],
        embeddings_model=cfg["embeddings"],
        persist_directory=cfg["presistent_directory"],
        chain_type="stuff",
        k=cfg["k"],
    )

    tools = [
        Tool(
            name="invoke",
            func=retrieval_chain.ainvoke,
            return_direct=True,
            description=(
                "invoke({{'query': query}}) -> str:\n"
                " - Executes a search for the query on vectorstore.\n"
                " - Returns the results of the RAG serach and the source documents.\n"
                " - `query`: query to search for on Wikipedia, e.g., What are the different kinds of pets people commonly own?."
            ),
            stringify_rule=lambda args: f"invoke({args[0]})",
        ),
    ]

    return tools
