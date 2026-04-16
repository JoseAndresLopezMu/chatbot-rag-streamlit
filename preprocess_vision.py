"""
preprocess_vision.py — Ejecutar localmente UNA VEZ antes del docker build.

Renderiza cada página de los PDFs como imagen y llama a Groq Vision para
extraer descripciones detalladas de tablas, gráficos e imágenes.
El resultado se guarda en documents/vision_descriptions.json y debe
hacerse commit para que el Docker build lo incluya en el índice.

Uso:
    pip install pymupdf groq python-dotenv
    GROQ_API_KEY=<tu_clave> python preprocess_vision.py
    git add documents/vision_descriptions.json
    git commit -m "docs: add Groq Vision descriptions for PDF pages"
    git push
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

try:
    import fitz
except ImportError:
    print("ERROR: instala pymupdf:  pip install pymupdf")
    sys.exit(1)

try:
    from groq import Groq
except ImportError:
    print("ERROR: instala groq sdk:  pip install groq")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
DPI_SCALE = 2.0  # ~144 dpi
SKIP_TOKEN = "SIN_CONTENIDO_VISUAL"

VISION_PROMPT = (
    "Eres un experto en análisis de documentos. "
    "Analiza esta página y describe con precisión todos los elementos visuales que encuentres: "
    "tablas (extrae TODOS los datos fila a fila), gráficos (valores y tendencias), "
    "imágenes informativas o diagramas. "
    "Si la página solo contiene texto plano sin elementos visuales relevantes, "
    f"responde únicamente: {SKIP_TOKEN}"
)


def render_page_b64(page: "fitz.Page") -> str:
    mat = fitz.Matrix(DPI_SCALE, DPI_SCALE)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def describe_page(client: "Groq", b64_image: str, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64_image}"
                }},
            ],
        }],
        max_tokens=768,
    )
    return response.choices[0].message.content.strip()


def process_pdfs(docs_dir: Path, model: str) -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    results: dict = {}

    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No se encontraron PDFs en {docs_dir}")
        return results

    print(f"Procesando {len(pdf_paths)} PDFs con modelo: {model}\n")

    for pdf_path in pdf_paths:
        fname = pdf_path.name
        print(f"[{fname}]")
        results[fname] = {}

        doc = fitz.open(str(pdf_path))
        for idx in range(len(doc)):
            page_num = idx + 1
            print(f"  Página {page_num}/{len(doc)} ... ", end="", flush=True)
            try:
                b64 = render_page_b64(doc[idx])
                desc = describe_page(client, b64, model)
                if SKIP_TOKEN in desc:
                    print("sin contenido visual")
                else:
                    results[fname][str(page_num)] = desc
                    print(f"OK ({len(desc)} chars)")
            except Exception as e:
                print(f"ERROR: {e}")
        doc.close()
        print()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesa PDFs con Groq Vision.")
    parser.add_argument("--docs-dir", default="./documents", help="Directorio de PDFs")
    parser.add_argument("--model", default=VISION_MODEL, help="Modelo Groq Vision a usar")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: define la variable de entorno GROQ_API_KEY")
        sys.exit(1)

    docs_dir = Path(args.docs_dir)
    output_path = docs_dir / "vision_descriptions.json"

    # Load existing descriptions to allow incremental runs
    existing: dict = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Cargadas descripciones existentes para: {list(existing.keys())}\n")

    new_results = process_pdfs(docs_dir, args.model)

    # Merge: existing is overwritten by new results
    merged = {**existing, **new_results}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in merged.values())
    print(f"Guardadas {total} descripciones en {output_path}")
    print("\nSiguiente paso:")
    print("  git add documents/vision_descriptions.json")
    print("  git commit -m 'docs: add Groq Vision descriptions'")
    print("  git push")


if __name__ == "__main__":
    main()
