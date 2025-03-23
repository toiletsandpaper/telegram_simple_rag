import asyncio
import logging
import os

import dotenv
from anyio import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from tg_rag_smolagents.utils import cleanup_file, parse_document, restrict_to_user_id

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the bot is started."""
    if not update.message:
        return
    logger.info("Received /start command from user: %s", update.effective_user)
    await update.message.reply_text("Hello! Send me a PDF file, and I'll process it for you.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    if not update.message:
        return
    logger.info("Received /help command from user: %s", update.effective_user)
    await update.message.reply_text(
        "I can help you process PDF files. Just send me a file, and I'll take care of the rest!"
    )

@restrict_to_user_id
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming files, parse them, and load as LangChain documents."""
    if not update.message:
        return
    if update.message and not update.message.document:
        logger.warning("User sent a message without a document.")
        await update.message.reply_text("Please send a valid file.")
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
        await update.message.reply_text(
            f"File processed successfully! Here's a preview:\n\n```{docs[:2]}```\n\n"
        )
    except Exception as e:
        logger.error("Error while processing file: %s, Error: %s", file_path, e)
        await update.message.reply_text(f"An error occurred while processing the file: {e}")
    finally:
        await cleanup_file(file_path)
        logger.info("File cleaned up: %s", file_path)
        await update.message.reply_text("File cleaned up successfully.")

def main() -> None:
    """Start the bot."""
    asyncio.gather(UPLOAD_DIR.mkdir(parents=True, exist_ok=True))
    logger.info("Upload directory created: %s", UPLOAD_DIR)

    dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv(raise_error_if_not_found=True))
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable is not set.")
        return



    logger.info("Bot is starting...")
    try:
        app = ApplicationBuilder().token(bot_token).build()
        logger.info("Setting up command handlers...")
        app.add_handler(MessageHandler(filters.ALL, handle_file))
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))

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
