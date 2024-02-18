import os
from pathlib import Path
from tqdm import tqdm

import dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()


# function to load the text docs
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


# split the docs into chunks using recursive character splitter
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


if __name__ == "__main__":
    list_of_datasets = [
        "BlockchainSolanaDataset",
        "PaulGrahamEssayDataset",
        "BraintrustCodaHelpDeskDataset",
        "Uber10KDataset2021",
        "EvaluatingLlmSurveyPaperDataset",
        "Llama2PaperDataset",
        "OriginOfCovid19Dataset",
        "PatronusAIFinanceBenchDataset",
    ]
    # dataset_name = "BraintrustCodaHelpDeskDataset"

    for dataset_name in tqdm(list_of_datasets):
        directory = f"datasets/{dataset_name}/source_files"
        documents = load_docs(directory)
        len(documents)
        # store the splitte documnets in docs variable
        docs = split_docs(documents)

        # embeddings using langchain
        api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-ada-002",
        )

        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo-1106",
            temperature=0,
        )

        # using chromadb as a vectorstore and storing the docs in it
        persist_directory = f"resources/chroma_db/{dataset_name}"
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persist_directory
        )
        vectordb.persist()
