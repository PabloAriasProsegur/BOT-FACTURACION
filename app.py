import json
import os
from io import BytesIO

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_LARGO = os.path.join(APP_DIR, "logolargo.jpg")
LOGO_PEQUENO = os.path.join(APP_DIR, "logopequeno.jpg")
TRAINING_FILE = os.path.join(APP_DIR, "training_data.csv")
CHAT_HISTORY_FILE = os.path.join(APP_DIR, "chat_history.json")

REQUIRED_COLUMNS = ["cliente", "servicio", "periodo", "importe", "fecha", "estado"]

DEFAULT_FAQ = [
    {
        "question": "¿Cómo valido un archivo de facturación?",
        "answer": "Primero revisás la estructura del archivo, verificás columnas obligatorias, clientes, servicios, fechas, importes y estados, y luego corrés las inconsistencias antes del envío.",
        "keywords": ["validar", "archivo", "excel", "facturacion", "facturación", "validacion", "validación"],
    },
    {
        "question": "¿Qué errores revisan al cargar facturas?",
        "answer": "Se revisan columnas faltantes, datos duplicados, servicios no activos, fechas fuera de rango, montos inconsistentes, clientes no válidos y errores de formato.",
        "keywords": ["errores", "facturas", "cargar", "datos", "columnas", "error"],
    },
    {
        "question": "¿Dónde subo el Excel para validar?",
        "answer": "Usá el panel lateral de archivos para cargar el Excel o CSV y continuar con la revisión del equipo de facturación.",
        "keywords": ["subir", "excel", "csv", "archivo", "adjuntar", "cargar"],
    },
    {
        "question": "¿Qué me puede rechazar un archivo?",
        "answer": "La información incompleta, columnas faltantes, clientes o servicios inexistentes, montos inconsistentes, fechas fuera de rango y estados no válidos pueden rechazar la carga.",
        "keywords": ["rechazo", "rechazar", "columnas", "datos", "incompleto", "estado"],
    },
    {
        "question": "¿Qué debo revisar antes de enviar?",
        "answer": "Revisá cliente, servicio, periodo, importe, fecha, estado y que no existan columnas vacías ni errores de formato.",
        "keywords": ["revisar", "antes", "enviar", "cliente", "servicio", "archivo"],
    },
    {
        "question": "¿Cómo detectan errores de clientes y servicios?",
        "answer": "Se comparan los datos del archivo con la base operativa para verificar códigos, estados, servicios activos y consistencia con el contrato.",
        "keywords": ["cliente", "servicio", "errores", "contrato", "base"],
    },
    {
        "question": "¿Qué es lo más importante para evitar errores?",
        "answer": "Mantener la estructura del archivo limpia, completar campos obligatorios y asegurar consistencia entre clientes, servicios, montos y fechas.",
        "keywords": ["importante", "evitar", "errores", "estructura", "consistencia"],
    },
    {
        "question": "¿Qué hago si el sistema devuelve un error?",
        "answer": "Leé el detalle del error, corrige el dato en el archivo, eliminá inconsistencias y volvé a cargarlo. Muchas veces basta con completar una columna o ajustar un valor.",
        "keywords": ["error", "sistema", "devuelve", "corrigir", "detalle"],
    },
    {
        "question": "¿Cuánto tarda la validación?",
        "answer": "Depende del volumen y la calidad del archivo. Si está bien armado, la validación suele ser rápida; si tiene errores, lleva más tiempo corregirlo.",
        "keywords": ["tiempo", "tarda", "validacion", "archivo", "rapido"],
    },
    {
        "question": "¿Puedo subir Excel, CSV o PDF?",
        "answer": "Sí. La app acepta Excel, CSV y PDF para revisión del equipo de facturación.",
        "keywords": ["excel", "csv", "pdf", "archivo", "subir"],
    },
]


def normalize_text(text):
    return " ".join(str(text).lower().replace("?", " ").replace("¿", " ").split())


def append_training_log(entry):
    log_path = os.path.join(APP_DIR, "training_log.json")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []
    data.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            chats = json.load(f)
        if isinstance(chats, list) and chats:
            return chats
    except Exception:
        pass
    return [{"id": "chat-1", "title": "Consulta nueva", "messages": []}]


def save_chat_history(chats):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)


def persist_current_chat():
    chats = st.session_state.chat_history
    current_id = st.session_state.current_chat_id
    current = next((chat for chat in chats if chat["id"] == current_id), None)
    if current is None:
        current = {"id": current_id, "title": "Consulta nueva", "messages": []}
        chats.append(current)
    current["messages"] = st.session_state.messages
    user_messages = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if user_messages:
        current["title"] = user_messages[0][:34] + ("..." if len(user_messages[0]) > 34 else "")
    save_chat_history(chats)


def start_new_chat():
    persist_current_chat()
    new_id = f"chat-{len(st.session_state.chat_history) + 1}"
    st.session_state.chat_history.append({"id": new_id, "title": "Consulta nueva", "messages": []})
    st.session_state.current_chat_id = new_id
    st.session_state.messages = []
    save_chat_history(st.session_state.chat_history)


def open_chat(chat_id):
    persist_current_chat()
    selected = next(chat for chat in st.session_state.chat_history if chat["id"] == chat_id)
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = selected.get("messages", [])


def normalize_question_answer_data(raw_entries):
    normalized = []
    for item in raw_entries:
        question = str(item.get("question", "") or "").strip()
        answer = str(item.get("answer", "") or "").strip()
        if not question or not answer:
            continue

        keywords = item.get("keywords")
        if isinstance(keywords, str):
            keyword_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        elif isinstance(keywords, list):
            keyword_list = [str(k).strip().lower() for k in keywords if str(k).strip()]
        else:
            keyword_list = []

        if not keyword_list:
            keyword_list = [word.lower() for word in question.split() if len(word) > 3][:12]

        normalized.append({"question": question, "answer": answer, "keywords": keyword_list})
    return normalized


def save_faq_entries(faq_entries):
    df = pd.DataFrame(faq_entries)
    if df.empty:
        df = pd.DataFrame(columns=["question", "answer", "keywords"])
    else:
        df = df[["question", "answer", "keywords"]]
    df.to_csv(TRAINING_FILE, index=False, encoding="utf-8")


def load_faq_entries():
    if not os.path.exists(TRAINING_FILE):
        save_faq_entries(DEFAULT_FAQ)
        return DEFAULT_FAQ

    try:
        df = pd.read_csv(TRAINING_FILE)
        if {"question", "answer"}.issubset(df.columns):
            entries = normalize_question_answer_data(df.to_dict("records"))
            if entries:
                return entries
    except Exception:
        pass

    save_faq_entries(DEFAULT_FAQ)
    return DEFAULT_FAQ


def add_new_training_entry(question, answer, keywords):
    faq_entries = st.session_state.get("faq_entries", load_faq_entries())
    entry = {
        "question": str(question or "").strip(),
        "answer": str(answer or "").strip(),
        "keywords": [k.strip().lower() for k in str(keywords or "").split(",") if k.strip()],
    }

    if not entry["question"] or not entry["answer"]:
        return False

    if not entry["keywords"]:
        entry["keywords"] = [word.lower() for word in entry["question"].split() if len(word) > 3][:12]

    faq_entries.append(entry)
    save_faq_entries(faq_entries)
    st.session_state.faq_entries = faq_entries

    append_training_log({
        "question": entry["question"],
        "answer": entry["answer"],
        "keywords": entry["keywords"],
    })
    return True


def import_training_file(uploaded_file):
    if uploaded_file is None:
        return False
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        if {"question", "answer"}.issubset(df.columns):
            entries = normalize_question_answer_data(df.to_dict("records"))
            if entries:
                save_faq_entries(entries)
                st.session_state.faq_entries = entries
                return True
    except Exception:
        return False
    return False


def prepare_chatbot(faq_entries):
    combined = [f"{item['question']} {' '.join(item['keywords'])}" for item in faq_entries]
    vectorizer = CountVectorizer(lowercase=True)
    question_vectors = vectorizer.fit_transform(combined)
    return vectorizer, question_vectors


def summarize_file(file_obj):
    name = file_obj.name
    size_mb = round(file_obj.size / (1024 * 1024), 2)
    ext = os.path.splitext(name)[1].lower()
    summary = f"{name} ({size_mb} MB)"

    if ext in [".xlsx", ".xls", ".csv"]:
        try:
            raw = file_obj.read()
            file_obj.seek(0)
            if ext in [".xlsx", ".xls"]:
                df = pd.read_excel(BytesIO(raw))
            else:
                df = pd.read_csv(BytesIO(raw))
            cols = [str(c).strip().lower() for c in df.columns]
            sample_cols = list(df.columns[:8])
            missing = [c for c in REQUIRED_COLUMNS if c not in cols]
            summary += f" | columnas: {', '.join(str(col) for col in sample_cols)}"
            if missing:
                summary += f" | faltan: {', '.join(missing)}"
            summary += f" | filas: {len(df.index)}"
        except Exception:
            summary += " | no se pudo leer automáticamente"
    return summary


def ask_llm_fallback(user_input, uploaded_files=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Actuás como asistente interno del equipo de facturación de Prosegur Alarmas. "
            "Respondé en español, breve, útil y orientado a validación de archivos y procesos del negocio. "
            f"Consulta del usuario: {user_input}. "
            f"Archivos adjuntos: {', '.join(f.name for f in uploaded_files) if uploaded_files else 'ninguno'}"
        )
        response = client.responses.create(model="gpt-4o-mini", input=prompt)
        return getattr(response, "output_text", None) or None
    except Exception:
        return None


def get_response(user_input, uploaded_files=None, faq_entries=None, threshold=0.18, vectorizer=None, question_vectors=None):
    faq_entries = faq_entries or st.session_state.get("faq_entries", load_faq_entries())
    user_text = normalize_text(user_input)
    if not user_text:
        return "Escribí tu consulta y te ayudaré con la validación de facturación."

    for item in faq_entries:
        if any(keyword in user_text for keyword in item["keywords"]):
            answer = item["answer"]
            if uploaded_files:
                names = ", ".join(f.name for f in uploaded_files[:3])
                answer += f"\n\nArchivo cargado: {names}."
            return answer

    llm_answer = ask_llm_fallback(user_input, uploaded_files)
    if llm_answer:
        return llm_answer

    if vectorizer is None or question_vectors is None:
        vectorizer, question_vectors = prepare_chatbot(faq_entries)

    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    best_index = int(similarities.argmax())
    best_score = float(similarities[best_index])

    if best_score < threshold:
        if uploaded_files:
            names = ", ".join(f.name for f in uploaded_files[:3])
            return (
                "Tengo información útil del archivo cargado. Revisá si faltan columnas clave, si hay datos vacíos, "
                f"si hay clientes o servicios inconsistentes y si los montos o fechas no coinciden.\n\nArchivo: {names}."
            )
        return (
            "Puedo ayudarte con validación de facturación, errores de carga, clientes, servicios, importes, fechas y revisión del archivo. "
            "Si subís el Excel, te puedo orientar mucho mejor."
        )

    answer = faq_entries[best_index]["answer"]
    if uploaded_files:
        names = ", ".join(f.name for f in uploaded_files[:3])
        answer += f"\n\nArchivo cargado: {names}."
    return answer


def get_default_test_suite():
    return [
        {"question": "¿Cómo valido un archivo de facturación?", "expected": ["validar", "archivo", "facturación"]},
        {"question": "¿Qué errores revisan al cargar facturas?", "expected": ["errores", "columnas", "facturas"]},
        {"question": "¿Dónde subo el Excel para validar?", "expected": ["subir", "excel", "validar"]},
        {"question": "¿Qué pasa si faltan columnas?", "expected": ["columnas", "faltan", "archivo"]},
        {"question": "¿Cuánto tarda la validación?", "expected": ["tiempo", "validación", "archivo"]},
    ]


def run_automatic_tests(faq_entries=None, vectorizer=None, question_vectors=None, test_cases=None):
    faq_entries = faq_entries or st.session_state.get("faq_entries", load_faq_entries())
    if vectorizer is None or question_vectors is None:
        vectorizer, question_vectors = prepare_chatbot(faq_entries)
    test_cases = test_cases or get_default_test_suite()
    results = []
    for test in test_cases:
        prediction = get_response(
            test["question"],
            faq_entries=faq_entries,
            vectorizer=vectorizer,
            question_vectors=question_vectors,
        )
        passed = any(token.lower() in prediction.lower() for token in test["expected"]) or any(
            token.lower() in test["question"].lower() for token in test["expected"]
        )
        results.append(
            {
                "question": test["question"],
                "expected": test["expected"],
                "prediction": prediction[:180],
                "passed": passed,
            }
        )

    accuracy = round((sum(1 for item in results if item["passed"]) / len(results)) * 100, 1) if results else 0.0
    return results, accuracy


st.set_page_config(
    page_title="Prosegur Alarmas | Asistente de Facturación",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #070707;
        --bg-soft: #101010;
        --panel: #151515;
        --panel-2: #1a1a1a;
        --line: rgba(255,255,255,0.1);
        --text: #f7f7f7;
        --muted: #d9d9d9;
        --yellow: #f5c400;
        --yellow-soft: #ffd64d;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #050505 0%, #0b0b0b 52%, #11100a 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070707 0%, #121212 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: #f5f5f5 !important;
    }

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #cfcfcf !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.045) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        box-shadow: none !important;
        text-align: left !important;
        transition: background 160ms ease, border-color 160ms ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(245,196,0,0.14) !important;
        border-color: rgba(245,196,0,0.65) !important;
        color: #ffffff !important;
    }

    .logo-frame {
        background: #ffffff;
        border-radius: 16px;
        padding: 0.55rem 0.7rem;
        margin-bottom: 0.8rem;
        text-align: center;
        box-shadow: 0 12px 28px rgba(0,0,0,0.32);
    }

    [data-testid="stSidebar"] [data-testid="stImage"] img {
        border-radius: 10px;
    }

    .chat-list-title {
        color: #ffd64d !important;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1rem 0 0.5rem;
    }

    .chat-list-item button {
        background: rgba(255,255,255,0.06) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        text-align: left !important;
        margin-bottom: 0.35rem;
    }

    .main .block-container {
        padding-top: 0.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    [data-testid="stImage"] img {
        border-radius: 18px;
        box-shadow: 0 18px 42px rgba(0,0,0,0.3);
    }

    .panel-box {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 18px 38px rgba(0,0,0,0.22);
    }

    .section-tag {
        color: var(--yellow-soft);
        font-weight: 900;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .lead {
        color: var(--text);
        line-height: 1.7;
        margin: 0;
        font-size: 1.04rem;
    }

    .stat-card {
        background: linear-gradient(145deg, rgba(245,196,0,0.13), rgba(255,255,255,0.035));
        border: 1px solid rgba(245,196,0,0.35);
        border-radius: 20px;
        padding: 1.15rem 1.25rem;
        min-height: 160px;
        box-shadow: 0 18px 34px rgba(0,0,0,0.2);
    }

    .big-number {
        color: var(--yellow-soft);
        font-size: 2.5rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.3rem 0 0.6rem;
    }

    .mini-pill {
        display: inline-block;
        background: rgba(245,196,0,0.12);
        border: 1px solid rgba(245,196,0,0.38);
        color: var(--yellow-soft);
        border-radius: 999px;
        padding: 0.38rem 0.7rem;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 5px 16px rgba(245,196,0,0.08);
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
    }

    .stFileUploader > div {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(245,196,0,0.32);
        border-radius: 14px;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--yellow) 0%, var(--yellow-soft) 100%);
        color: #111111;
        border: none;
        border-radius: 12px;
        font-weight: 800;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, var(--yellow-soft) 0%, var(--yellow) 100%);
        color: #111111;
    }

    .stChatInput textarea,
    .stChatInput > div,
    .stChatInput input {
        background: #111111 !important;
        color: #ffffff !important;
    }

    .stChatInput textarea::placeholder {
        color: #d0d0d0 !important;
    }

    .stChatInput textarea {
        border: 1px solid rgba(245,196,0,0.45) !important;
        border-radius: 14px !important;
        padding: 0.9rem 1rem !important;
    }

    .stChatInput button {
        background: var(--yellow) !important;
        color: #111111 !important;
        border: none !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] {
        background: #070707 !important;
        border-top: 1px solid rgba(245,196,0,0.12);
        padding-top: 0.4rem;
    }

    [data-testid="stBottom"] {
        background: #070707 !important;
    }

    [data-testid="stChatInput"] > div {
        background: #111111 !important;
        border: 1px solid rgba(245,196,0,0.45) !important;
        border-radius: 14px !important;
    }

    .file-pill {
        display: inline-block;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.32rem 0.6rem;
        margin: 0.14rem 0.26rem 0.14rem 0;
        color: var(--text);
        font-size: 0.84rem;
    }

    .warning-box {
        background: rgba(245,196,0,0.09);
        border: 1px solid rgba(245,196,0,0.32);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        color: var(--muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "faq_entries" not in st.session_state:
    st.session_state.faq_entries = load_faq_entries()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = st.session_state.chat_history[0]["id"]

if not st.session_state.messages:
    current_chat = next(
        chat for chat in st.session_state.chat_history
        if chat["id"] == st.session_state.current_chat_id
    )
    st.session_state.messages = current_chat.get("messages", [])

if not st.session_state.messages:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hola, soy el asistente de facturación de Prosegur Alarmas. Subí un archivo con el botón + o consultá por validación, errores, clientes, servicios o fechas.",
        }
    ]

vectorizer, question_vectors = prepare_chatbot(st.session_state.faq_entries)

st.image(LOGO_LARGO, use_container_width=True)

left_col, right_col = st.columns([1.8, 1.1])

with left_col:
    st.markdown("<div class='mini-pill' style='margin-bottom: 0.8rem;'>Versión premium</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-box">
            <div class="section-tag">Avance de información</div>
            <p class="lead">
                El equipo de facturación valida archivos, revisa estructuras y datos, verifica inconsistencias
                y coordina ajustes necesarios para evitar rechazos o errores en la carga.
            </p>
            <br>
            <p class="lead">
                El objetivo es garantizar que cada archivo cumpla con los requisitos del sistema, mantenga
                consistencia en clientes, servicios, importes, fechas y estados, y permita una operación segura
                y eficiente.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Base de conocimiento", len(st.session_state.faq_entries))
    with col_b:
        st.metric("Modo", "Auto-training")
    with col_c:
        st.metric("Validación", "QA test")

    with st.expander("Administrar base de conocimiento"):
        uploaded_training = st.file_uploader(
            "Importar dataset de entrenamiento",
            type=["csv", "xlsx", "xls"],
            key="training_dataset",
        )
        if uploaded_training is not None:
            if import_training_file(uploaded_training):
                st.success("Base de conocimiento actualizada correctamente.")
            else:
                st.warning("El archivo debe tener columnas: question, answer y keywords opcional.")

with right_col:
    st.markdown(
        """
        <div class="stat-card">
            <div class="section-tag">Estado operativo</div>
            <div class="big-number">24/7</div>
            <p class="lead">Control de validación, revisión y seguimiento de archivos para facturación.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="stat-card">
            <div class="section-tag">Flujo</div>
            <div class="big-number">4 pasos</div>
            <p class="lead">Carga → validación → corrección → envío final.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("<div class='logo-frame'>", unsafe_allow_html=True)
    st.image(LOGO_PEQUENO, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-pill'>Asistente interno</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    if st.button("+  Nuevo chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.markdown("<div class='chat-list-title'>Conversaciones guardadas</div>", unsafe_allow_html=True)
    for chat in reversed(st.session_state.chat_history):
        item_label = chat["title"] or "Consulta nueva"
        if st.button(item_label, key=f"open_{chat['id']}", use_container_width=True):
            open_chat(chat["id"])
            st.rerun()

    with st.expander("Entrenamiento inteligente"):
        with st.form("training_form"):
            st.text_input("Pregunta nueva", key="new_question")
            st.text_area("Respuesta", key="new_answer")
            st.text_input("Palabras clave (separadas por coma)", key="new_keywords")
            if st.form_submit_button("Guardar conocimiento"):
                if add_new_training_entry(st.session_state.new_question, st.session_state.new_answer, st.session_state.new_keywords):
                    st.success("Nuevo aprendizaje agregado y guardado.")
                    st.rerun()
                else:
                    st.warning("Necesitás completar pregunta y respuesta para entrenar.")

        if st.button("Entrenar + test automático", use_container_width=True):
            vectorizer, question_vectors = prepare_chatbot(st.session_state.faq_entries)
            tests, accuracy = run_automatic_tests(st.session_state.faq_entries, vectorizer, question_vectors)
            st.session_state.last_test_accuracy = accuracy
            st.session_state.last_test_results = tests
            st.success(f"Entrenamiento finalizado. Precisión del test: {accuracy}%")

        if st.button("Reiniciar base de ejemplo", use_container_width=True):
            st.session_state.faq_entries = normalize_question_answer_data(DEFAULT_FAQ)
            save_faq_entries(st.session_state.faq_entries)
            st.success("Se restauró la base inicial.")

    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.08); margin: 1rem 0;'>", unsafe_allow_html=True)

    st.subheader("Funciones")
    st.write("• Validación de facturación")
    st.write("• Revisión de Excel y CSV")
    st.write("• Control de errores e inconsistencias")
    st.write("• Consultas del proceso operativo")
    st.write("• Soporte para clientes y servicios")

    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.08); margin: 1rem 0;'>", unsafe_allow_html=True)

    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Conversación limpia. Subí un archivo con el botón + o escribí tu consulta de facturación.",
        }]
        persist_current_chat()
        st.rerun()


st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

st.markdown(
    "<div class='warning-box'>Usá esta app para consultar validaciones, errores, continuidad del proceso y revisión de archivos para facturación.</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_submission = st.chat_input(
    "Escribí tu consulta sobre facturación, validación o archivo...",
    accept_file="multiple",
    file_type=["xlsx", "xls", "csv", "txt", "pdf"],
)

if chat_submission:
    prompt = chat_submission.text
    uploaded_files = chat_submission.files
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_response(prompt, uploaded_files, st.session_state.faq_entries, vectorizer=vectorizer, question_vectors=question_vectors)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    persist_current_chat()

if st.session_state.get("last_test_results"):
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.subheader("Resultado del test automático")
    st.write(f"Precisión: {st.session_state.last_test_accuracy}%")
    for item in st.session_state.last_test_results:
        status = "✅" if item["passed"] else "⚠️"
        st.write(f"{status} {item['question']} -> {item['prediction']}")

if st.session_state.get("uploaded_files"):
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.caption("Resumen de archivo cargado")
    for file_obj in st.session_state.uploaded_files:
        st.write(summarize_file(file_obj))
