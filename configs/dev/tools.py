from src.agents.tools import Tool
from src.tools.knowledge_tools.react_retrieval import ReActRetrieval
from src.tools.internet_tools.google_search import GoogleSearch


def generate_tools(cfg):
    tools = []
    for persist_directory in cfg["presistent_directory"]:
        retrieval_chain = ReActRetrieval.setup_retrieval_system(
            llm_model=cfg["default_model"],
            embeddings_model=cfg["embeddings"],
            persist_directory=persist_directory,
            chain_type="stuff",
            k=cfg["rtv_k"],
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
                f" - Then executes a search for the query within the specified collection name {collection_name} in the vectorstore."
                f" - The collection will only contain the information from the {collection_name}\n"
                "(for example, if the collection_name is llama2_dataset, then the collection only contains documents about llama2, the '_dataset' is only the suffix)\n"
                " - It returns the results of the RAG serach and the corresponding source documents.\n"
                " - `query`: query to search for on vector database, e.g., What are the different kinds of pets people commonly own?."
            ),
            stringify_rule=lambda args: f"{tool_name}({args[0]})",
        )
        tools.append(tool)

    # Add google search tool
    google_search = GoogleSearch(k=cfg["ggs_k"])
    tool = Tool(
        name="google_search",
        # func=google_search.search_client.run,
        func=google_search.asearch_top_k,
        return_direct=True,
        description=(
            "google_search(query: str) -> str:\n"
            " - This function takes a query as input."
            " - Then executes a search for the query using google search API."
            " - It returns the top k results for the query.\n"
            " - `query`: query to search for on google, e.g., What are the different kinds of pets people commonly own?."
        ),
        stringify_rule=lambda args: f"google_search({args[0]})",
    )
    tools.append(tool)

    return tools
