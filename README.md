# 🤖 Chatbot de Atención al Público

Aplicación sencilla desarrollada con **Python**, **scikit-learn** y **Streamlit**.

El chatbot compara la pregunta escrita por el usuario con un conjunto de preguntas frecuentes mediante:

- `CountVectorizer`
- similitud del coseno

y devuelve la respuesta correspondiente a la pregunta más parecida.

## Estructura del proyecto

```text
chatbot_streamlit/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Ejecutar localmente

Crea un entorno virtual si lo deseas e instala las dependencias:

```bash
pip install -r requirements.txt
```

Después ejecuta:

```bash
streamlit run app.py
```

## Subir a GitHub

1. Crea un repositorio nuevo en GitHub.
2. Sube `app.py`, `requirements.txt`, `README.md` y `.gitignore`.
3. No necesitas subir el notebook para que funcione la aplicación; puedes conservarlo en el repositorio solo como material de desarrollo.

## Publicar en Streamlit Community Cloud

1. Ingresa a Streamlit Community Cloud e inicia sesión con GitHub.
2. Selecciona **Create app**.
3. Elige el repositorio donde subiste estos archivos.
4. Selecciona la rama principal, normalmente `main`.
5. En **Main file path**, escribe:

```text
app.py
```

6. Pulsa **Deploy**.

Streamlit instalará automáticamente las librerías indicadas en `requirements.txt`.

## Personalizar las preguntas

Edita las listas `QUESTIONS` y `ANSWERS` de `app.py`. Cada pregunta debe ocupar la misma posición que su respuesta.

Ejemplo:

```python
QUESTIONS = [
    "¿Cuál es el horario de atención?",
    "¿Dónde está ubicada la tienda?"
]

ANSWERS = [
    "Atendemos de lunes a viernes...",
    "Estamos ubicados en..."
]
```

## Nota

Este chatbot es un sistema basado en similitud de texto y preguntas frecuentes. No utiliza un LLM ni una API externa, por lo que no necesita API keys.
