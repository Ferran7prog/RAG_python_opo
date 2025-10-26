# 🧠 Sistema RAG con LangChain, Supabase y OpenAI

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** simple utilizando **Python**, **LangChain**, **Supabase**, **Flask** y modelos de lenguaje de **OpenAI**.

---

## 🚀 Descripción

El objetivo del proyecto es permitir consultas inteligentes sobre una base de conocimiento personalizada.  
El sistema combina recuperación de información (a través de *embeddings* en Supabase) con generación de respuestas usando un modelo de lenguaje (LLM) de OpenAI.

---

## 🧩 Tecnologías principales

- **Python** — lenguaje principal del proyecto  
- **Flask** — framework web para crear la API  
- **LangChain** — orquestación entre recuperación y generación  
- **Supabase** — base de datos vectorial para almacenar y buscar *embeddings*  
- **OpenAI API** — modelo LLM para generar respuestas contextuales  

---

## ⚙️ Instalación

1. **Clona este repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/tu-repo.git
   cd tu-repo
   ``` 

Crea y activa un entorno virtual:
```
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

Instala las dependencias:
```
pip install -r requirements.txt
```
🔑 Variables de entorno

Crea un archivo .env con las siguientes variables:
```
OPENAI_API_KEY=tu_clave_openai
SUPABASE_URL=tu_url_supabase
SUPABASE_KEY=tu_api_key_supabase
```
▶️ Ejecución

Inicia el servidor Flask:
```
python app.py
```

La API quedará disponible en http://localhost:5000.

🧠 Uso

Envía una consulta al endpoint, por ejemplo:
```
curl -X POST http://localhost:5000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "¿Qué es LangChain?"}'
```

El sistema recuperará los documentos más relevantes y generará una respuesta contextual.

🧰 Estructura básica del proyecto
```
.
├── app.py                     # Lógica principal del RAG y Flask
├── app-text uploader.ipynb    # Subir contenido vectorizado a la base de datos
├── requirements.txt           # Dependencias
└── README.md
```

