import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from utils import create_and_persist_vectorstore

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PDF_DIRECTORY = "./documents"
CHUNK_SIZE = 1000
CHUNK_TAG = "customv2"
TAGS_FOR_METADATA = ["general", "formulario"]
INDEX_NAME_FILE = Path("last_index.txt")
VISUALIZE_CHUNKS = True

if __name__ == "__main__":
    logger.info("Iniciando el proceso de indexación de documentos...")
    PDF_PATHS = [str(p) for p in Path(PDF_DIRECTORY).glob("*.pdf")]
    if not PDF_PATHS:
        logger.error(f"No se encontraron archivos PDF en el directorio '{PDF_DIRECTORY}'.")
        sys.exit(1)

    try:
        _, index_name = create_and_persist_vectorstore(
            embedding_model_name=EMBEDDING_MODEL_NAME,
            pdf_paths=PDF_PATHS,
            chunk_size=CHUNK_SIZE,
            chunk_tag=CHUNK_TAG,
            tags_for_metadata=TAGS_FOR_METADATA,
            collection_name_prefix="rag_docs_advanced",
            pdf_folder_path=PDF_DIRECTORY.replace("./", ""),
            include_image_descriptions=True,
            visualize_chunks=VISUALIZE_CHUNKS
        )

        with open(INDEX_NAME_FILE, "w") as f:
            f.write(index_name)

        logger.success(f"Indexación completada. El índice se guardó como: '{index_name}'")
        logger.info(f"El nombre del índice se ha guardado automáticamente en '{INDEX_NAME_FILE}'.")

    except Exception as e:
        logger.error(f"Error durante el proceso de indexación: {e}", exc_info=True)
        sys.exit(1)
