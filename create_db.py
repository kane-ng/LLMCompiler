from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import dotenv

dotenv.load_dotenv()

directory = "datasets/dev"


# function to load the text docs
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


documents = load_docs(directory)
len(documents)


# split the docs into chunks using recursive character splitter
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


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
persist_directory = "resources/chroma_db"
vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)
vectordb.persist()
