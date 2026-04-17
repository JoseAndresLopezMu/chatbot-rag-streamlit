import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser

from utils import (
    extract_structured_info_from_pdf,
)

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FORMULARIO_PDF_PATH = "./documents/formulario.pdf"
INDEX_NAME_FILE = Path("last_index.txt")

PROMPTS_DIR = Path("./prompts")
try:
    PYTHON_PROMPT_TEMPLATE = (PROMPTS_DIR / "calculator.prompt.txt").read_text(encoding="utf-8")
    RAG_SYSTEM_PROMPT = (PROMPTS_DIR / "rag_system.prompt.txt").read_text(encoding="utf-8")
except FileNotFoundError:
    st.error("Error: La carpeta 'prompts' o los archivos de prompts no se encontraron.")
    st.stop()

def load_index_name() -> str | None:
    try:
        return INDEX_NAME_FILE.read_text().strip()
    except FileNotFoundError:
        return None

@st.cache_resource
def load_llm():
    return ChatGroq(model=GROQ_MODEL, temperature=0)

@st.cache_resource
def load_retriever():
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    index_name = load_index_name()

    # Auto-build index on first run (e.g. Streamlit Cloud where filesystem is ephemeral)
    if not index_name:
        pdf_paths = [str(p) for p in Path("./documents").glob("*.pdf")]
        if not pdf_paths:
            st.error("No se encontraron documentos PDF en la carpeta 'documents/'.", icon="❌")
            st.stop()
        with st.spinner("Indexando documentos (esto solo ocurre la primera vez)..."):
            from utils import create_and_persist_vectorstore
            _, index_name = create_and_persist_vectorstore(
                embedding_model_name=EMBEDDING_MODEL_NAME,
                pdf_paths=pdf_paths,
                chunk_size=1000,
                chunk_tag="cloud",
                tags_for_metadata=["general"],
                collection_name_prefix="rag_docs",
                pdf_folder_path="documents",
                include_image_descriptions=False,
                visualize_chunks=False,
            )
            INDEX_NAME_FILE.write_text(index_name)

    try:
        vector_store = Chroma(
            persist_directory=f"./chroma_db/{index_name}",
            embedding_function=embedding_model,
            collection_name=index_name,
        )
        st.success(f"Base de datos '{index_name}' cargada.", icon="✅")
        return vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.1, "k": 4},
        )
    except Exception as e:
        st.error(f"Error al cargar la base de datos vectorial: {e}", icon="❌")
        st.stop()

def execute_python_code(code: str) -> str:
    try:
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            exec(code)  # nosec B102
        return f"El resultado es: {f.getvalue()}"
    except Exception as e:
        return f"Error ejecutando código: {e}"

def get_formatted_form_data() -> str:
    data = st.session_state.get("extracted_form_data", {})
    if data:
        response = "He encontrado los siguientes datos en el formulario:\n\n"
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').capitalize()
            response += f"- **{formatted_key}:** {value}\n"
        return response
    return "No se encontraron datos estructurados en el archivo formulario.pdf."

def get_rag_chain(_llm, _retriever):
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Dada la conversación, reescribe la pregunta de seguimiento para que sea una pregunta independiente."),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(_llm, _retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(_llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain

st.set_page_config(page_title="Asistente Inteligente de Documentos", page_icon="📄")
st.title("📄 Asistente Inteligente de Documentos")

llm = load_llm()
retriever = load_retriever()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=800, return_messages=True, memory_key="chat_history", output_key='answer'
    )
memory = st.session_state.memory

if "form_data_loaded" not in st.session_state:
    if os.path.exists(FORMULARIO_PDF_PATH):
        st.info("Detectado 'formulario.pdf', extrayendo datos estructurados...")
        data = extract_structured_info_from_pdf(FORMULARIO_PDF_PATH)
        st.session_state.extracted_form_data = data
        if data:
            st.success("Datos del formulario cargados.", icon="✅")
        else:
             st.warning("Se encontró 'formulario.pdf' pero no se pudieron extraer datos.", icon="⚠️")
    else:
        st.session_state.extracted_form_data = {}
    st.session_state.form_data_loaded = True

router_prompt = ChatPromptTemplate.from_template(
    """Dada la pregunta del usuario, clasifícala como 'python' o 'rag'.
- Usa 'python' para preguntas que requieran un cálculo matemático o de código.
- Usa 'rag' para todas las demás preguntas.

Devuelve una única palabra: python o rag.

Pregunta: {input}"""
)
router_chain = router_prompt | llm | StrOutputParser()

python_prompt = ChatPromptTemplate.from_template(PYTHON_PROMPT_TEMPLATE)
python_chain = python_prompt | llm | StrOutputParser() | execute_python_code

rag_chain = get_rag_chain(llm, retriever)
form_chain = RunnablePassthrough.assign(answer=lambda x: get_formatted_form_data())

with st.sidebar:
    st.header("Acciones Especiales")
    if st.button("Mostrar Datos del Formulario"):
        if os.path.exists(FORMULARIO_PDF_PATH):
            with st.spinner("Extrayendo datos..."):
                extracted_data = extract_structured_info_from_pdf(FORMULARIO_PDF_PATH)
                if extracted_data:
                    st.success("Datos extraídos del formulario:")
                    st.json(extracted_data)
                else:
                    st.error("No se pudieron extraer datos del formulario.")
        else:
            st.error(f"No se encuentra el archivo: {FORMULARIO_PDF_PATH}")

chat_history = memory.load_memory_variables({}).get("chat_history", [])
for message in chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

if prompt := st.chat_input("Haz una pregunta sobre los documentos o pide un cálculo..."):
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Clasificando pregunta..."):
            route = router_chain.invoke({"input": prompt})

        answer = ""
        source_docs = []

        if "python" in route.lower():
            st.info("Ruta seleccionada: **Código Python**", icon="🐍")
            with st.spinner("Ejecutando cálculo..."):
                answer = python_chain.invoke({"input": prompt})
            st.markdown(answer)

        else:
            st.info("Ruta seleccionada: **Búsqueda en Documentos (RAG)**", icon="📚")
            placeholder = st.empty()
            accumulated = ""

            for chunk in rag_chain.stream({"input": prompt, "chat_history": chat_history}):
                if "context" in chunk:
                    source_docs = chunk["context"]
                if "answer" in chunk:
                    accumulated += chunk["answer"]
                    placeholder.markdown(accumulated + "▌")

            placeholder.markdown(accumulated)
            answer = accumulated

            # Source citations
            if source_docs:
                seen: set = set()
                unique_sources = []
                for doc in source_docs:
                    key = (
                        doc.metadata.get("file_name", "?"),
                        doc.metadata.get("page_number", "?"),
                    )
                    if key not in seen:
                        seen.add(key)
                        unique_sources.append(doc)
                with st.expander("📎 Ver fuentes consultadas", expanded=False):
                    for doc in unique_sources:
                        fname = doc.metadata.get("file_name", "Desconocido")
                        pnum = doc.metadata.get("page_number", "?")
                        st.markdown(f"- **{fname}** — Página {pnum}")

        if answer:
            memory.save_context({"input": prompt}, {"answer": answer})

if st.button("Limpiar Chat"):
    st.session_state.clear()
    st.rerun()
