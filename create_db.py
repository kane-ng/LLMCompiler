from __future__ import annotations
import os
from pathlib import Path
from tqdm import tqdm

import dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_directory_docs(
    path: str, multithreading: bool = True, verbose: bool = True
) -> list[Document]:
    loader = DirectoryLoader(
        path=path, show_progress=verbose, use_multithreading=multithreading
    )
    docs: list[Document] = loader.load()
    return docs


def recursive_document_chunking(
    docs: list[Document], chunk_size: int = 512, chunk_overlap: int = 128
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs: list[Document] = text_splitter.split_documents(documents=docs)
    return split_docs


if __name__ == "__main__":
    list_of_datasets = [
        "10k_uber_2021_dataset",
        "braintrust_coda_dataset",
        "origin_of_covid19_dataset",
        "paul_graham_essay_dataset",
        "eval_llm_survey_paper_dataset",
        "gemini_paper_dataset",
        "llama2_paper_dataset",
        "viact_dataset",
        "blockchain_solana_dataset",
        "gpt_paper_dataset",
        "patronus_financebench_dataset",
    ]
    # dataset_name = "viact_dataset"

    for dataset_name in list_of_datasets:
        print(f"Processing {dataset_name} dataset")

        # load the documents from the directory
        directory: str = f"datasets/{dataset_name}/source_files"
        documents: list[Document] = load_directory_docs(path=directory)
        split_docs: list[Document] = recursive_document_chunking(docs=documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        # embeddings using langchain
        embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002",
            # allowed_special={"<|endofprompt|>"},
            disallowed_special=(),
        )

        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo-1106",
            temperature=0,
        )

        # using chromadb as a vectorstore and storing the docs in it
        persist_directory = f"resources/chroma_db/{dataset_name}"
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        vectordb.persist()
        print(f"There are {vectordb._collection.count()} documents in the vectorstore")
        print("-" * 80)
