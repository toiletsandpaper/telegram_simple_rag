import logging
import os
from functools import wraps

import dotenv
from anyio import Path
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

#i_f = InputFormat()

async def parse_document(file_path: str | Path) -> list[Document]:
    docs = await DoclingLoader(file_path=str(file_path)).aload()
    return docs

async def cleanup_file(file_path: Path) -> None:
    await file_path.unlink(missing_ok=True)


def restrict_to_user_id(func):  # type: ignore
    dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv(raise_error_if_not_found=True))
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
