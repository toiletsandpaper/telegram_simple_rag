from anyio import Path
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader

#i_f = InputFormat()

async def parse_document(file_path: str | Path) -> list[Document]:
    """
    Parse a document using docling-parse.

    Args:
        file_path (str): Path to the file to be parsed.

    Returns:
        str: Extracted text content from the document.

    Raises:
        ValueError: If the file type is unsupported or parsing fails.
    """
    docs = await DoclingLoader(file_path=str(file_path)).aload()
    return docs



async def cleanup_file(file_path: Path) -> None:
    """
    Delete the file after processing.

    Args:
        file_path (Path): Path to the file to be deleted.
    """
    await file_path.unlink(missing_ok=True)
