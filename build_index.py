"""Run during Docker build to pre-build the ChromaDB index."""
from pathlib import Path
from utils import create_and_persist_vectorstore

pdf_paths = [str(p) for p in Path("./documents").glob("*.pdf")]
if not pdf_paths:
    raise RuntimeError("No PDFs found in documents/")

_, index_name = create_and_persist_vectorstore(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    pdf_paths=pdf_paths,
    chunk_size=1000,
    chunk_tag="cloud",
    tags_for_metadata=["general"],
    collection_name_prefix="rag_docs",
    pdf_folder_path="documents",
    include_image_descriptions=False,
    visualize_chunks=False,
)
Path("last_index.txt").write_text(index_name)
print("Index built:", index_name)
