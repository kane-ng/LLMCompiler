import logging
from typing import List, Optional, Type

import chromadb
from chromadb.api.models.Collection import Collection
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma

from utils.config import load_db_config

config = load_db_config()
logger = logging.getLogger(__name__)


class ChromaService(Chroma):
    HOST: str = config["chroma_db"]["host"]
    PORT: int = config["chroma_db"]["port"]
    embedding_func: any = OpenAIEmbeddings()
    client = (
        chromadb.HttpClient(
            host=HOST,
            port=PORT,
        )
        if HOST and PORT
        else None
    )
    if client is None:
        logger.warning("No client found! Using local client")

    @classmethod
    def list_collection(cls) -> list[Collection]:
        colls = cls.client.list_collections()
        return colls

    @classmethod
    def create_collection(cls, collection_name: str) -> Collection:
        collection = cls.client.create_collection(
            name=collection_name, embedding_function=cls.embedding_func
        )
        return collection

    @classmethod
    def get_collection(cls, collection_name: str) -> Collection:
        collection = cls.client.get_collection(
            name=collection_name, embedding_function=cls.embedding_func
        )
        return collection

    @classmethod
    def delete_collection(cls, collection_name: str) -> None | ValueError:
        cls.client.delete_collection(collection_name)

    @classmethod
    def create_vectorstore(
        cls: Type["ChromaService"],
        list_documents: List[Document],
        embedding_model: Optional[Embeddings],
        collection_name: str,
        persist_directory: Optional[str],
    ) -> Chroma:
        """
        Create a Chroma vectorstore from a list of documents.
        If client not found, default to use local client
        """
        documents = cls.from_documents(
            list_documents,
            embedding_model,
            collection_name=collection_name,
            client=cls.client,
            persist_directory=persist_directory,
        )
        return documents

    @classmethod
    def load_vectorstore(
        cls: Type["ChromaService"],
        collection_name: str,
        persist_directory: str | None = None,
        embedding_function: Embeddings | None = None,
    ) -> Chroma:
        """
        Load an existing Chroma vectorstore.
        If client not found, default to use local client
        """
        vectorstore = cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
            client=cls.client,
            embedding_function=embedding_function
            if embedding_function
            else cls.embedding_func,
        )
        return vectorstore

    @classmethod
    def query(cls, query: str) -> List[Document]:
        docs = cls.similarity_search(query=query)
        return docs

    @staticmethod
    def naive_chunk_files(
        file_path: str, chunk_size: int = 1000, chunk_overlap: int = 0
    ) -> List[Document]:
        """
        Upload a single file -> Return a list of documents.
        """
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        list_documents = text_splitter.split_documents(documents)

        return list_documents

    @staticmethod
    def naive_chunk_directory(
        folder_path: str,
        glob: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        multithreading: bool = False,
    ) -> List[Document]:
        """
        Upload a single folder of files -> Return a list of documents.
        """
        loader = DirectoryLoader(
            folder_path, glob=glob, use_multithreading=multithreading
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        list_documents = text_splitter.transform_documents(documents)

        return list_documents

    @classmethod
    def healthcheck(cls):
        return cls.client.heartbeat()
