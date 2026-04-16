import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser

from utils import (
    extract_structured_info_from_pdf,
)

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
EMBEDDING_MODEL_NAME = "nomic-embed-text"
FORMULARIO_PDF_PATH = "./documents/formulario.pdf"
INDEX_NAME_FILE = Path("last_index.txt")

PROMPTS_DIR = Path("./prompts")
try:
    PYTHON_PROMPT_TEMPLATE = (PROMPTS_DIR / "calculator.prompt.txt").read_text()
    RAG_SYSTEM_PROMPT = (PROMPTS_DIR / "rag_system.prompt.txt").read_text()
except FileNotFoundError:
    st.error("Error: La carpeta 'prompts' o los archivos de prompts no se encontraron.")
    st.stop()

def load_index_name() -> str | None:
    try:
        return INDEX_NAME_FILE.read_text().strip()
    except FileNotFoundError:
        return None

INDEX_TO_LOAD = load_index_name()

@st.cache_resource
def load_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)

@st.cache_resource
def load_retriever():
    if not INDEX_TO_LOAD:
        st.error("Error: No se encontró el índice. Por favor, ejecuta el script `index.py` primero.", icon="❌")
        st.stop()

    try:
        from langchain_community.embeddings import OllamaEmbeddings
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        vector_store = Chroma(
            persist_directory=f"./chroma_db/{INDEX_TO_LOAD}",
            embedding_function=embedding_model,
            collection_name=INDEX_TO_LOAD
        )
        st.success(f"Base de datos '{INDEX_TO_LOAD}' cargada.", icon="✅")
        return vector_store.as_retriever()
    except Exception as e:
        st.error(f"Error al cargar la base de datos vectorial '{INDEX_TO_LOAD}': {e}", icon="❌")
        st.stop()

def execute_python_code(code: str) -> str:
    try:
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            exec(code)
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
        with st.spinner("Pensando..."):
            route = router_chain.invoke({"input": prompt})
            answer = ""

            if "python" in route.lower():
                st.info("Ruta seleccionada: **Código Python**", icon="🐍")
                answer = python_chain.invoke({"input": prompt})
            else:
                st.info("Ruta seleccionada: **Búsqueda en Documentos (RAG)**", icon="📚")
                response = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
                answer = response.get("answer", "No pude encontrar una respuesta.")

            st.markdown(answer)

            memory.save_context({"input": prompt}, {"answer": answer})

if st.button("Limpiar Chat"):
    st.session_state.clear()
    st.rerun()
