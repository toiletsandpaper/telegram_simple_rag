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
    """Parse a document from the given file path."""
    logger.info("Parsing document from file: %s", file_path)
    docs = await DoclingLoader(file_path=str(file_path)).aload()
    logger.info("Document parsing completed.")
    return docs


def configure_qdrant() -> tuple[QdrantClient, QdrantVectorStore]:
    """Configure and return a Qdrant client and vector store."""
    logger.info("Configuring Qdrant client and vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={'truncation': True, 'max_length': 512}
    )
    embeddings.embed_query("test")  # Warm up the model

    client = QdrantClient(path='/tmp/langchain_qdrant')

    # Create collection if it doesn't exist
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    except ValueError:
        logger.info("Collection already exists. Skipping creation.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    logger.info("Qdrant configuration completed.")
    return client, vector_store


async def update_store(vector_store: QdrantVectorStore, docs: list[Document]) -> dict[str, Document]:
    """Update the vector store with the given documents."""
    logger.info("Splitting documents into chunks...")
    splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=512,
        chunk_overlap=50,
        model_name="BAAI/bge-m3"
    )
    splitted_docs = splitter.split_documents(docs)
    logger.info("Document splitting completed.")

    ids = [str(uuid4()) for _ in splitted_docs]
    logger.info("Adding documents to vector store...")
    await vector_store.aadd_documents(documents=splitted_docs, ids=ids)
    logger.info("Documents successfully added to vector store.")

    return {ids[i]: splitted_docs[i] for i in range(len(ids))}


def configure_retriever(
    vector_store: QdrantVectorStore,
    search_type: str = "mmr",
    search_kwargs: dict[str, Any] = {"k": 5}
):
    """Configure and return a retriever for the vector store."""
    logger.info("Configuring retriever with search type: %s and search kwargs: %s", search_type, search_kwargs)
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


async def cleanup_file(file_path: Path) -> None:
    """Delete the specified file if it exists."""
    logger.info("Cleaning up file: %s", file_path)
    await file_path.unlink(missing_ok=True)
    logger.info("File cleanup completed.")


def restrict_to_user_id(func):  # type: ignore
    """Decorator to restrict access to a specific user ID."""
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
