from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
api_key = "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"
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


# vectordb = Chroma(
#     persist_directory=str(persist_directory),
#     embedding_function=embeddings,
# )
retriever = vectordb.as_retriever(search_kwargs={"k": 7})

retrieval_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)


# Doing similarity search  using query
query = "What are the different kinds of pets people commonly own?"
result = retrieval_chain.invoke({"query": query})
result
