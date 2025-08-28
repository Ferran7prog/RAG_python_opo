from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Importar todas las librerias y funciones necesarias
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from supabase import create_client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")

TABLE_NAME = "documents"
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)


OPEN_AI_MODEL = "gpt-4o-mini"
TOP_K = 2

# Crear modelo de embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Para la base de datos vectorial
def get_vector_store() -> SupabaseVectorStore:
    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase_client,
        table_name=TABLE_NAME,
        query_name=None,
    )

# Para extraer la info relevante de la vector store en base a la similitud con la query
def retrieve(query: str, k: int=TOP_K) -> list[str]:
    vs = get_vector_store()
    results = vs.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


vectorstore = get_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# Prompt para el sistema que siempre lo tenga en cuenta y orden de el promt a pasar al modelo llm
system_prompt = (
    """You are a Strict RAG Assistant. Answer only based on Retrieved Context.
    If the answer is not in the context, say "No forma parte del temario"."""
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context: \n {context} \n\n Question: {question}"),
])

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cadena
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Crear Flask app
app = Flask(__name__)
CORS(app)  

# Ruta para servir al fichero HTML
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Ruta para obtener las preguntas que vienen del frontend 
@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        # Pregunta del frontend
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        print(f"Received question: {question}")  
        
        # Invocando la cadena, para obtener resultados
        result = rag_chain.invoke(question)
        
        print(f"Generated answer: {result}")  
        
        # Enviar la respuesta de vuelta al frontend 
        return jsonify({'answer': result})
        
    except Exception as e:
        print(f"Error: {str(e)}")  
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Para correr el servidor
if __name__ == '__main__':
    print("Starting server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)