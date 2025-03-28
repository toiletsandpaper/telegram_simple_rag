import logging
import os
from functools import wraps
from typing import Any
from uuid import uuid4

from anyio import Path
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tg-store")

async def parse_document(file_path: str | Path) -> list[Document]:
    docs = await DoclingLoader(file_path=str(file_path)).aload()
    return docs

def configure_qdrant() -> tuple[QdrantClient, QdrantVectorStore]:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={'truncation': True, 'max_length': 512})
    embeddings.embed_query("test")  # warm up the model


    client = QdrantClient(
        path='/tmp/langchain_qdrant'
    )

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    except ValueError:
        pass

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return client, vector_store

async def update_store(vector_store: QdrantVectorStore, docs: list[Document]) ->  dict[str, Document]:
    splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=512,
        chunk_overlap=50,
        model_name="BAAI/bge-m3"
    )
    logger.info("Splitting documents into chunks...")
    splitted_docs = splitter.split_documents(docs)
    logger.info("Splitting completed.")
    ids = [str(uuid4()) for _ in enumerate(splitted_docs)]
    logger.info("Adding documents to vector store...")
    await vector_store.aadd_documents(documents=splitted_docs, ids=ids)
    logger.info("Documents added to vector store.")
    return {ids[i]: splitted_docs[i] for i in range(len(ids))}

def configure_retriever(vector_store: QdrantVectorStore, search_type: str = "mmr", search_kwargs: dict[str, Any] = {"k": 5}):
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

async def cleanup_file(file_path: Path) -> None:
    await file_path.unlink(missing_ok=True)


def restrict_to_user_id(func):  # type: ignore
    AUTHORIZED_USER_ID = os.getenv("ALLOWED_USER_ID")
    if not AUTHORIZED_USER_ID:
        logger.error("ALLOWED_USER_ID environment variable is not set.")
        raise ValueError("ALLOWED_USER_ID environment variable is not set.")
    @wraps(func)  # type: ignore
    def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):  # type: ignore
        if not update.effective_user:
            logger.warning("No effective user found in the update.")
            return
        if str(update.effective_user.id) != AUTHORIZED_USER_ID:
            logger.warning("Unauthorized access attempt by user: %s", update.effective_user.id)
            return
        return func(update, context, *args, **kwargs)  # type: ignore
    return wrapped  # type: ignore
