import asyncio
import logging
import os

import dotenv
from anyio import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from tg_rag_smolagents.utils import (
    cleanup_file,
    configure_qdrant,
    configure_retriever,
    parse_document,
    restrict_to_user_id,  # type: ignore
    update_store,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv(raise_error_if_not_found=True))

CLIENT, VECTOR_STORE = configure_qdrant()
RETIREVER = configure_retriever(vector_store=VECTOR_STORE)

UPLOAD_DIR = Path("uploads")

MODEL_NAME = 'qwen2-7b-instruct'
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tg-store")

uuids: list[str] = []

@restrict_to_user_id
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming files, parse them, and load as LangChain documents."""
    global uuids
    if not update.message:
        return

    if update.message.text and update.message.text.startswith("RESET"):
        try:
            logger.info("Resetting the vector store...")
            print(COLLECTION_NAME)
            if uuids and VECTOR_STORE.delete(uuids): # type: ignore
                uuids = []
                logger.info("Vector store reset successfully.")
                await update.message.reply_text("Vector store reset successfully.")
            else:
                logger.error("Failed to reset the vector store.")
                await update.message.reply_text("Failed to reset the vector store.")
        except Exception as e:
            logger.error("Error while resetting the vector store: %s", e)
            await update.message.reply_text(f"An error occurred while resetting the vector store: {e}")
        finally:
            return

    if update.message and not update.message.document:
        user_message = update.message.text
        if not user_message:
            await update.message.reply_text("Please send a valid text message.")
            return

        try:
            logger.info("Retrieving documents for user query: %s", user_message)
            retrieved_docs = RETIREVER.invoke(user_message)
            if not retrieved_docs:
                await update.message.reply_text("No relevant documents found for your query.")
                return

            logger.info("Asking OpenAI LLM about the retrieved documents...")

            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_api_base_url = os.getenv("OPENAI_API_BASE_URL")
            if not openai_api_base_url:
                logger.error("OPENAI_API_BASE_URL environment variable is not set.")
                await update.message.reply_text("OpenAI API base URL is missing. Please configure it.")
                return
            if not openai_api_key:
                logger.error("OPENAI_API_KEY environment variable is not set.")
                await update.message.reply_text("OpenAI API key is missing. Please configure it.")
                return

            model = ChatOpenAI(
                base_url=openai_api_base_url,
                model=MODEL_NAME
            )


            messages = [  # type: ignore
                SystemMessage("You are a helpful assistant."),
                HumanMessage(f"Based on these documents: {retrieved_docs}, answer the query: {user_message}")
            ]
            response = await model.ainvoke(messages)  # type: ignore
            await update.message.reply_text(f"LLM's response: {response.text()}")
        except Exception as e:
            logger.error("Error while processing user query: %s, Error: %s", user_message, e)
            await update.message.reply_text(f"An error occurred while processing your query: {e}")
        return

    file = update.message.document
    if not file:
        logger.warning("No document found in the message.")
        await update.message.reply_text("Please send a valid file.")
        return

    if not file.file_name:
        logger.warning("Received a file without a name.")
        await update.message.reply_text("File name is missing.")
        return

    file_path = UPLOAD_DIR / str(file.file_name)
    file_data = await file.get_file()
    logger.info("Received file: %s from user: %s", file.file_name, update.effective_user)
    await update.message.reply_text(f"File '{file.file_name}' received. Processing...")
    await file_data.download_to_drive(str(file_path))

    try:
        logger.info("Parsing file: %s", file_path)
        docs = await parse_document(file_path)
        logger.info("File parsed successfully: %s", file_path)
        uuid_docs_mapping = await update_store(VECTOR_STORE, docs)
        await update.message.reply_text("File processed successfully! Here's a preview:\n")
        for uuid, doc in uuid_docs_mapping.items():
            await update.message.reply_text(f"- {uuid}: {doc.page_content[:100]}...\n")
            break
        logger.info("Documents added to vector store: %s", uuid_docs_mapping)
        uuids.extend(uuid_docs_mapping.keys()) # type: ignore
    except Exception as e:
        logger.error("Error while processing file: %s, Error: %s", file_path, e)
        await update.message.reply_text(f"An error occurred while processing the file: {e}")
        await cleanup_file(file_path)
        logger.info("File cleaned up: %s", file_path)
        await update.message.reply_text("File cleaned up successfully.")


def main() -> None:
    """Start the bot."""
    asyncio.gather(UPLOAD_DIR.mkdir(parents=True, exist_ok=True))
    logger.info("Upload directory created: %s", UPLOAD_DIR)
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable is not set.")
        return

    logger.info("Bot is starting...")
    try:
        app = ApplicationBuilder().token(bot_token).build()
        logger.info("Setting up command handlers...")
        app.add_handler(MessageHandler(filters.ALL, handle_message))

        if not app.updater:
            logger.critical("Updater is not set. Exiting...")
            return
        logger.info("Polling...")
        app.run_polling(
            drop_pending_updates=True,
            timeout=10,
        )
    except Exception as e:
        logger.critical("Error while starting the bot: %s", e)
    finally:
        logger.info("Bot is shutting down...")

if __name__ == "__main__":
    main()
