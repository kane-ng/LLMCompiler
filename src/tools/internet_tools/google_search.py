from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper


class GoogleSearch:
    def __init__(self, k: int):
        self.k = k
        self.search_client = GoogleSearchAPIWrapper()

    async def asearch_top_k(self, query: str) -> str:
        return self.search_client.results(query, self.k)
