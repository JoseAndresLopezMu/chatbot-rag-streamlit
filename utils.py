import os
from pathlib import Path
from typing import Tuple, List
from uuid import uuid4
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from loguru import logger
from tqdm import tqdm
import re
from datetime import datetime
import pymupdf4llm
from cache_to_disk import cache_to_disk
from chonkie import RecursiveChunker, RecursiveLevel, RecursiveRules
try:
    from chonkie.chunker.base import BaseChunker
except ImportError:
    BaseChunker = object  # type: ignore[assignment,misc]
try:
    from chonkie.utils import Visualizer
except ImportError:
    Visualizer = None  # type: ignore[assignment,misc]

try:
    import easyocr
    import fitz
    import io
except ImportError:
    logger.warning("Dependencias de OCR no encontradas. La funcionalidad estará deshabilitada.")
    easyocr, fitz, io = None, None, None

file_path = os.path.realpath(__file__)
root_dir = Path(file_path).parent

@cache_to_disk(1)
def get_md_from_pdf_path(pdf_path: str) -> str:
    md = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
    md = re.sub(r"\n-{3,}\n", "\n\n", md)
    md = re.sub(r"={5,} Page \d+ ={5,}\n", "", md)
    md = re.sub(r"\n\*\*([^\n]+)\*\*\s*\n", r"\n## \1\n\n", md)
    md = re.sub(r"\n(\d+\.)\s*\*\*([^\n]+)\*\*", r"\n\1 \2", md)
    return md

def extract_metadata_advanced(pdf_path: str, base_index_dir: Path, chunks_text: list[str]) -> list[dict]:
    stat = os.stat(pdf_path)
    file_name = os.path.basename(pdf_path)
    return [{
        "file_name": file_name,
        "full_pdf_path": pdf_path,
        "file_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "chunk_index": idx,
    } for idx, _ in enumerate(chunks_text)]

def get_chunker_advanced() -> BaseChunker:
    return RecursiveChunker(
        chunk_size=1000,
        rules=RecursiveRules(levels=[
            RecursiveLevel(delimiters=["\n\n"], whitespace=False),
            RecursiveLevel(delimiters=["\n"], whitespace=False),
        ]),
        min_characters_per_chunk=50,
        return_type="chunks",
    )

def create_chunk_documents(pdf_paths: list[str], include_image_descriptions: bool, visualize_chunks: bool = False) -> List[Document]:
    all_documents = []
    chunker = get_chunker_advanced()
    visualizer = Visualizer() if Visualizer is not None else None

    for pdf_path in tqdm(pdf_paths, desc="Procesando PDFs"):
        # 1. Obtenemos el texto base del PDF (sin OCR)
        md_text = get_md_from_pdf_path(pdf_path)

        # 2. Si se quiere visualizar, lo hacemos AHORA, solo con el texto del PDF
        if visualize_chunks and visualizer is not None:
            logger.info("Generando visualización de chunks (solo texto de PDF)...")
            # El visualizador necesita los chunks Y el texto original como tercer argumento posicional
            chunks_for_viz = chunker(md_text)
            output_path = f"visualizacion_{Path(pdf_path).stem}.html"
            visualizer.save(chunks_for_viz, output_path, md_text)
            logger.success(f"Visualización guardada en '{output_path}'.")

        # 3. Ahora, añadimos el texto del OCR (si corresponde) para la indexación final
        if include_image_descriptions:
            image_descs = get_image_descriptions(pdf_path)
            if image_descs:
                md_text += "\n\n--- Descripciones de Imágenes (OCR) ---\n" + "\n".join(image_descs)

        # 4. Generamos los chunks definitivos a partir del texto completo (PDF + OCR)
        final_chunks = chunker(md_text)

        # 5. Creamos los documentos para la base de datos a partir de los chunks definitivos
        metadata_list = extract_metadata_advanced(pdf_path, root_dir, [c.text for c in final_chunks])
        for i, chunk in enumerate(final_chunks):
            all_documents.append(Document(page_content=chunk.text, metadata=metadata_list[i]))

    return all_documents

def get_vectorstore_from_disk(embedding_model_name: str, index_name: str) -> Chroma:
    db_path = root_dir / f"chroma_db/{index_name}"
    if not db_path.exists():
        raise FileNotFoundError(f"El índice '{index_name}' no existe. Ejecuta index.py primero.")

    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = Chroma(
        persist_directory=str(db_path),
        embedding_function=embedding_model,
        collection_name=index_name
    )
    logger.success(f"Base de datos vectorial '{index_name}' cargada.")
    return vector_store

def create_and_persist_vectorstore(embedding_model_name: str, pdf_paths: List[str], **kwargs) -> Tuple[Chroma, str]:
    index_name = f"{kwargs.get('collection_name_prefix', 'rag_docs')}-{uuid4()}"
    db_path = root_dir / f"chroma_db/{index_name}"

    # Pasamos el nuevo parámetro a la función que crea los documentos
    visualize = kwargs.get('visualize_chunks', False)
    documents = create_chunk_documents(
        pdf_paths,
        kwargs.get('include_image_descriptions', False),
        visualize_chunks=visualize
    )

    logger.info(f"Creando índice '{index_name}' con {len(documents)} chunks...")

    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(db_path),
        collection_name=index_name
    )
    vector_store.persist()
    logger.success(f"Índice guardado en {db_path}")
    return vector_store, index_name

def extract_structured_info_from_pdf(pdf_path: str) -> dict:
    extracted_data = {}
    try:
        md_text = get_md_from_pdf_path(pdf_path)
        lines = md_text.split('\n')

        table_headers = []
        is_in_table_data = False

        for line in lines:
            cleaned_line = line.strip()

            if cleaned_line.startswith("|") and cleaned_line.endswith("|"):
                columns = [c.replace("<br>", " ").strip() for c in cleaned_line[1:-1].split("|")]

                if len(columns) == 2:
                    for cell in columns:
                        if ":" in cell:
                            key, value = map(str.strip, cell.split(":", 1))
                            extracted_data[key.lower().replace(" ", "_")] = value
                    continue

                if "nombre" in cleaned_line.lower() and "apellido1" in cleaned_line.lower():
                    table_headers = [h.lower().replace(" ", "_") for h in columns]
                    is_in_table_data = True
                    continue

                if is_in_table_data and not all(re.match(r"[-: ]*$", c) for c in columns if c):
                    for i, header in enumerate(table_headers):
                        if i < len(columns) and columns[i]:
                            final_header = f"empleado_{header}"
                            extracted_data[final_header] = columns[i]
                    is_in_table_data = False

            elif ":" in cleaned_line:
                key, value = map(str.strip, cleaned_line.split(":", 1))
                if key and value:
                    extracted_data[key.lower().replace(" ", "_")] = value

    except Exception as e:
        logger.error(f"Error final al procesar el formulario: {e}")

    return extracted_data

def get_image_descriptions(pdf_path: str) -> List[str]:
    if not all([easyocr, fitz, io]):
        return []
    try:
        reader = easyocr.Reader(['es'])
    except Exception as e:
        logger.error(f"No se pudo inicializar EasyOCR: {e}")
        return []
    image_texts = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            for img_info in doc.get_page_images(page_num):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    result = reader.readtext(image_bytes)
                    text = ' '.join([item[1] for item in result]).strip()
                    if text:
                        logger.info(f"OCR (Pág {page_num+1}): {text}")
                        image_texts.append(f"Texto de imagen en pág {page_num + 1}: {text}")
                except Exception as e:
                    logger.warning(f"Error en OCR para imagen en pág {page_num + 1}: {e}")
        doc.close()
    except Exception as e:
        logger.error(f"Error procesando imágenes de {pdf_path}: {e}")
    return image_texts
