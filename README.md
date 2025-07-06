# Chatbot RAG con Agente y Ejecución de Código 🚀

Este proyecto implementa un chatbot avanzado que utiliza un modelo de lenguaje (LLM) a través de **Ollama**. El chatbot es capaz de responder preguntas basándose en una colección de documentos PDF (técnica RAG), mantener el contexto de la conversación, ejecutar código Python y extraer información estructurada de formularios.

---

## ✨ Características Principales

* **Retrieval-Augmented Generation (RAG)**: Indexa documentos PDF en una base de datos vectorial (ChromaDB) para responder preguntas basadas en su contenido.
* **Agente con Herramientas (Agent with Tools)**: Utiliza un agente de LangChain que puede decidir de forma autónoma si necesita:
    * Buscar en los documentos (`search_documents`).
    * Ejecutar código Python para cálculos o lógica (`execute_python_code`).
    * Consultar datos extraídos de un formulario (`get_extracted_form_data`).
* **OCR para Imágenes**: Extrae texto de las imágenes contenidas en los PDF durante la indexación para enriquecer el contexto.
* **Memoria Dinámica**: Mantiene un historial de la conversación y lo resume automáticamente para no exceder el límite de tokens del LLM.
* **Interfaz Interactiva**: Construido con **Streamlit** para una experiencia de chat amigable.

---

## 🔧 Requisitos Previos

Antes de empezar, asegúrate de tener instalado el siguiente software en tu sistema:

1.  **Python**: Versión 3.10 o superior.
2.  **Ollama**: La aplicación de Ollama debe estar [instalada y en ejecución](https://ollama.com/). Es la responsable de servir los modelos de lenguaje localmente.

---

## ⚙️ Configuración e Instalación

Sigue estos pasos para configurar el proyecto en tu máquina local.

### Crear y Activar un Entorno Virtual
# Descargar el git
Descargar el código del git.

# Crear el entorno
uv sync

## Descargar modelo de Ollama
# Descargar el modelo de embeddings
ollama pull nomic-embed-text

# Descargar el modelo de lenguaje (ej. gemma3:4b o el que prefieras)
ollama pull gemma3:4b

## Ejecutar proyecto
1. uv run index.py (creación del index para ponerlo en app.py)
2. uv run python -m streamlit run app.py