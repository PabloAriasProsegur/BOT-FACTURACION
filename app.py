import os
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

st.set_page_config(
    page_title="Prosegur Alarmas | Facturación",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --black: #050505;
        --black-soft: #111111;
        --black-panel: #171717;
        --white: #f5f5f5;
        --muted: #d5d5d5;
        --yellow: #f4c300;
        --yellow-soft: #ffd649;
        --line: rgba(255, 255, 255, 0.12);
        --shadow: rgba(0,0,0,0.28);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #121212 100%);
        color: var(--white);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d0d 0%, #171717 100%);
        border-right: 1px solid var(--line);
    }

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .prosegur-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        background: #f4f4f4;
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }

    .prosegur-mark {
        width: 74px;
        height: 74px;
        border-radius: 50%;
        border: 4px solid #111111;
        background: #111111;
        position: relative;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .prosegur-mark::before {
        content: "";
        width: 44px;
        height: 44px;
        border-radius: 50%;
        border: 3px solid #f5f5f5;
        position: absolute;
        box-sizing: border-box;
    }

    .prosegur-mark::after {
        content: "";
        width: 22px;
        height: 22px;
        background: #f5f5f5;
        border-radius: 50%;
        position: absolute;
        box-shadow: 0 0 0 5px #111111;
    }

    .prosegur-title {
        color: #111111;
        font-size: clamp(2rem, 3vw, 3.2rem);
        font-weight: 900;
        line-height: 0.9;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .prosegur-subtitle {
        color: #111111;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        margin-top: 0.5rem;
        text-transform: uppercase;
    }

    .promo-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.15);
    }

    .label {
        color: var(--yellow-soft);
        font-weight: 800;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .text-muted {
        color: var(--muted);
        line-height: 1.7;
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(244,195,0,0.15), rgba(255,255,255,0.03));
        border: 1px solid rgba(244,195,0,0.35);
        border-radius: 16px;
        padding: 1rem;
    }

    .metric-card strong {
        display: block;
        color: var(--yellow-soft);
        font-size: 1.8rem;
        line-height: 1;
        margin-bottom: 0.3rem;
    }

    .chat-shell {
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.9rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.2);
    }

    .stChatMessage {
        border-radius: 14px;
    }

    .stFileUploader > div {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(244,195,0,0.32);
        border-radius: 14px;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--yellow) 0%, #ffd84d 100%);
        color: #111111;
        border: none;
        border-radius: 12px;
        font-weight: 800;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ffd84d 0%, var(--yellow) 100%);
        color: #111111;
    }

    .stDownloadButton > button {
        background: #111111;
        border: 1px solid rgba(244,195,0,0.32);
        color: var(--white);
    }

    .aplicativo-badge {
        display: inline-block;
        background: rgba(244,195,0,0.15);
        color: var(--yellow-soft);
        border: 1px solid rgba(244,195,0,0.4);
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


FAQ_ITEMS = [
    {
        "question": "¿Cómo valido un archivo de facturación?",
        "answer": "Primero revisá que el archivo tenga todas las columnas obligatorias, luego validá cliente, servicio, importe, fechas y estado. Si hay inconsistencias, corrégelo y vuelvelo a cargar.",
        "keywords": ["validar", "archivo", "excel", "facturacion", "facturación", "validacion", "validación"],
    },
    {
        "question": "¿Qué errores revisan al cargar facturas?",
        "answer": "Se revisan columnas faltantes, datos duplicados, clientes inexistentes, servicios sin cobertura, fechas fuera de rango, montos inconsistentes y estados no válidos.",
        "keywords": ["error", "errores", "cargar", "facturas", "excel", "datos", "columnas"],
    },
    {
        "question": "¿Dónde subo el Excel para validar?",
        "answer": "Usá el panel de adjuntos o el botón de carga de archivos de esta app para subir el Excel y continuar con la revisión del equipo de facturación.",
        "keywords": ["subir", "archivo", "excel", "adjuntar", "cargar", "upload"],
    },
    {
        "question": "¿Qué información necesito revisar antes de enviar el archivo?",
        "answer": "Revisá cliente, servicio, periodo, importe, vigencia, fechas, nombre del archivo y que no existan columnas vacías ni errores de formato.",
        "keywords": ["informacion", "revisar", "antes", "enviar", "archivo", "cliente", "servicio"],
    },
    {
        "question": "¿Qué pasa si faltan columnas o datos?",
        "answer": "El archivo puede quedar rechazado por validación. Hay que completar la información faltante, corregir el formato y volver a cargarlo.",
        "keywords": ["faltan", "columnas", "datos", "vacíos", "incompleto", "rechazado"],
    },
    {
        "question": "¿Cómo detectan errores de clientes y servicios?",
        "answer": "Se comparan los datos del Excel con la base operativa para verificar códigos, nombres, servicios activos e información asociada al contrato o al cliente.",
        "keywords": ["cliente", "servicio", "detectar", "errores", "base", "contrato"],
    },
    {
        "question": "¿Qué es lo más importante para evitar rechazos?",
        "answer": "Mantener la estructura del archivo limpia, completar los campos obligatorios y asegurar consistencia entre clientes, servicios, montos y fechas.",
        "keywords": ["rechazo", "rechazos", "evitar", "importante", "error", "validacion"],
    },
    {
        "question": "¿Cuáles son las validaciones principales del equipo?",
        "answer": "Las principales validaciones son: columnas obligatorias, formatos, montos, fechas, estado del servicio, duplicados, clientes y consistencia del archivo.",
        "keywords": ["validaciones", "principales", "equipo", "columnas", "montos", "fechas", "duplicados"],
    },
    {
        "question": "¿Qué hago si el sistema devuelve un error?",
        "answer": "Leé el detalle del error, corrige el dato en el archivo, elimina inconsistencias y vuelve a cargarlo. En muchos casos basta con completar una columna o corregir un valor.",
        "keywords": ["error", "sistema", "devuelve", "corrigir", "cargar", "detalle"],
    },
    {
        "question": "¿Cuánto tarda la validación?",
        "answer": "Depende del volumen y la calidad del archivo. Si el Excel está bien armado, la validación suele ser bastante rápida; si tiene errores, lleva más tiempo corregirlo.",
        "keywords": ["tiempo", "tarda", "validacion", "archivo", "rapido", "duracion", "duración"],
    },
    {
        "question": "¿Puedo subir un archivo de Excel o CSV?",
        "answer": "Sí. Esta app acepta Excel, CSV y archivos de texto para revisión del equipo de facturación.",
        "keywords": ["excel", "csv", "texto", "archivo", "subir"],
    },
    {
        "question": "¿Qué significa que un archivo tiene errores de validación?",
        "answer": "Significa que la información no cumple con la estructura o los requisitos mínimos necesarios para ser aceptada por el sistema de facturación.",
        "keywords": ["errores", "validacion", "significa", "archivo", "rechazo"],
    },
]


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().replace("?", " ").replace("¿", " ").split())


@st.cache_resource
def prepare_chatbot():
    combined = [f"{item['question']} {' '.join(item['keywords'])}" for item in FAQ_ITEMS]
    vectorizer = CountVectorizer(lowercase=True)
    question_vectors = vectorizer.fit_transform(combined)
    return vectorizer, question_vectors


vectorizer, question_vectors = prepare_chatbot()


def ask_llm_fallback(text: str, files=None) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Actuás como asistente interno de facturación de Prosegur Alarmas. "
            "Responde en español, breve y útil. "
            f"Consulta del usuario: {text}. "
            f"Archivos adjuntos: {', '.join(f.name for f in files) if files else 'ninguno'}."
        )
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return getattr(response, "output_text", None) or None
    except Exception:
        return None


def get_response(user_input: str, uploaded_files=None, threshold: float = 0.22) -> str:
    user_text = normalize_text(user_input)

    if not user_text:
        return "Escribí una consulta para que pueda ayudarte con validación de facturación."

    for item in FAQ_ITEMS:
        keywords = item["keywords"]
        if any(kw in user_text for kw in keywords):
            answer = item["answer"]
            if uploaded_files:
                names = ", ".join(file.name for file in uploaded_files[:3])
                answer += f"\n\nArchivo cargado: {names}."
            return answer

    llm_answer = ask_llm_fallback(user_input, uploaded_files)
    if llm_answer:
        return llm_answer

    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    best_index = int(similarities.argmax())
    best_score = float(similarities[best_index])

    if best_score < threshold:
        return (
            "No tengo una respuesta exacta para esa consulta, pero puedo ayudarte con validaciones del archivo, "
            "errores de carga, clientes, servicios, importes, fechas o revisión de facturación."
        )

    answer = FAQ_ITEMS[best_index]["answer"]
    if uploaded_files:
        names = ", ".join(file.name for file in uploaded_files[:3])
        answer += f"\n\nArchivo cargado: {names}."
    return answer


def render_logo():
    st.markdown(
        """
        <div class="prosegur-header">
            <div class="prosegur-mark"></div>
            <div>
                <div class="prosegur-title">PROSEGUR<br>ALARMS</div>
                <div class="prosegur-subtitle">Facturación</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_logo()

left_col, right_col = st.columns([1.9, 1.1])

with left_col:
    st.markdown(
        """
        <div class="promo-box">
            <div class="label">Avance de información</div>
            <p class="text-muted">
                El equipo de facturación validará archivos, verificará estructuras y datos, revisará inconsistencias
                y coordinará ajustes necesarios para evitar rechazos o errores en la carga.
            </p>
            <br>
            <p class="text-muted">
                El objetivo es garantizar que cada archivo cumpla con los requisitos del sistema, mantenga consistencia
                en clientes, servicios, importes, fechas y estados, y permita una operación más segura y eficiente.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown(
        """
        <div class="metric-card">
            <div class="label">Estado operativo</div>
            <strong>24/7</strong>
            <p class="text-muted">Control de validación, revisión y seguimiento de archivos para facturación.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="metric-card">
            <div class="label">Flujo</div>
            <strong>4 pasos</strong>
            <p class="text-muted">Carga → validación → corrección → envío final.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("<div class='aplicativo-badge'>Asistente interno</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.subheader("Funciones")
    st.write("• Validación de facturación")
    st.write("• Revisión de Excel y CSV")
    st.write("• Control de errores y inconsistencias")
    st.write("• Consultas del proceso operativo")
    st.write("• Soporte para clientes y servicios")

    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.08); margin: 1rem 0;'>", unsafe_allow_html=True)

    st.subheader("Adjuntar archivos")
    uploaded_files = st.file_uploader(
        "Seleccioná el archivo para revisar",
        type=["xlsx", "xls", "csv", "txt", "pdf"],
        accept_multiple_files=True,
        key="upload_files",
    )

    if uploaded_files:
        st.success(f"Se cargaron {len(uploaded_files)} archivo(s).")
        for file in uploaded_files[:5]:
            st.caption(f"- {file.name}")

    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.08); margin: 1rem 0;'>", unsafe_allow_html=True)

    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

st.subheader("📎 Carga de archivos y revisiones")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hola, soy el asistente de facturación de Prosegur Alarmas. Podés consultar validaciones, errores de carga, archivos adjuntos y flujo de revisión del equipo.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Escribí tu consulta sobre facturación, validación o archivo...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    uploaded_files = st.session_state.get("upload_files")
    response = get_response(prompt, uploaded_files)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

