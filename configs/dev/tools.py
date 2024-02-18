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

        collection_name = persist_directory.split("/")[-1]
        tool_name = f"{collection_name}_search".lower()
        tool = Tool(
            name=tool_name,
            func=retrieval_chain.ainvoke,
            return_direct=True,
            description=(
                f"{tool_name}({{'query': query}}) -> str:\n"
                " - This function takes a query as input."
                " - Then executes a search for the query within the specified collection name {collection_name} in the vectorstore."
                " - It returns the results of the RAG serach and the corresponding source documents.\n"
                " - `query`: query to search for on vector database, e.g., What are the different kinds of pets people commonly own?."
            ),
            stringify_rule=lambda args: f"{tool_name}({args[0]})",
        )
        tools.append(tool)

    return tools
