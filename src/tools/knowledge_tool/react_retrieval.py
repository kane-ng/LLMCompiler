import os
from pathlib import Path

import dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()


class ReActRetrieval:
    @staticmethod
    def setup_retrieval_system(
        llm_model: str,
        embeddings_model: str,
        persist_directory: Path,
        chain_type: str = "stuff",
        k: int = 5,
    ):
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=llm_model,
            temperature=0,
        )

        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=embeddings_model,
        )

        try:
            vectordb = Chroma(
                persist_directory=str(persist_directory),
                embedding_function=embeddings,
            )
            print("Loaded Collection from Chroma Vectorstore")

        except Exception as exc:
            raise RuntimeError(f"Error loading Chroma Vectorstore: {exc}")

        try:
            retriever = vectordb.as_retriever(search_kwargs={"k": k})
            retrieval_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=chain_type,
                retriever=retriever,
                return_source_documents=True,
            )
            print("Loaded Retrieval Chain")
            return retrieval_chain

        except Exception as exc:
            raise RuntimeError(f"Error loading Retrieval Chain: {exc}")
