import os, re, ast, json, time, unicodedata
import streamlit as st
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq

import os, streamlit as st
from dotenv import load_dotenv       # optional, for local dev

load_dotenv(override=False)          # pulls .env into os.environ when you run locally

def _get_secret(name: str) -> str | None:
    """Look in Streamlit-Cloud secrets first, then in OS env vars."""
    return st.secrets.get(name) or os.getenv(name)

SUPABASE_URL         = _get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = _get_secret("SUPABASE_SERVICE_KEY")
GROQ_API_KEY         = _get_secret("GROQ_API_KEY")


if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY]):
    st.error("Missing environment variables! Set them locally in .env "
             "or in Streamlit-Cloud Secrets.")
    st.stop()

MARKDOWN_TABLE_NAME  = "markdown_chunks"
ATTRIBUTE_TABLE_NAME = "Leoni_attributes"
RPC_FUNCTION_NAME    = "match_markdown_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
GROQ_MODEL_FOR_SQL   = "qwen-qwq-32b"
GROQ_MODEL_FOR_ANSWER = "qwen-qwq-32b"
VECTOR_SIMILARITY_THRESHOLD = 0.4
VECTOR_MATCH_COUNT   = 3

# ──────────────────────────────────────────────────────
# 2.   INITIALISE CLIENTS (once per session)
# ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_clients():
    sb  = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    groq_cli = Groq(api_key=GROQ_API_KEY)
    return sb, st_model, groq_cli

supabase, st_model, groq_client = _init_clients()

# ──────────────────────────────────────────────────────
# 3.  UTILITIES (unchanged except for print→st.write)
# ──────────────────────────────────────────────────────
def strip_think_tags(text: str) -> str:
    if not text: return text
    return re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>',
                  '', text, flags=re.IGNORECASE | re.DOTALL).strip()

def _normalise_chunk(row):
    if isinstance(row, dict) and "content" in row:
        return row
    if isinstance(row, dict) and len(row) == 1:
        row = next(iter(row.values()))
    if isinstance(row, str):
        try: row = json.loads(row)
        except json.JSONDecodeError:
            try: row = ast.literal_eval(row)
            except Exception:
                return {"content": row, "filename": "Unknown", "similarity": None}
    if isinstance(row, dict):
        row.setdefault("filename", "Unknown")
        row.setdefault("similarity", None)
        return row
    return {"content": str(row), "filename": "Unknown", "similarity": None}

def get_query_embedding(text):
    return st_model.encode(text).tolist() if text else None

def find_relevant_markdown_chunks(query_embedding):
    if not query_embedding: return []
    resp = supabase.rpc(
        RPC_FUNCTION_NAME,
        {
            'query_embedding': query_embedding,
            'match_threshold': VECTOR_SIMILARITY_THRESHOLD,
            'match_count': VECTOR_MATCH_COUNT
        }
    ).execute()
    return [_normalise_chunk(r) for r in (resp.data or [])]

def generate_sql_from_query(user_query, table_schema):
    prompt = f"""Your task is to … (same giant prompt as before) …
User Question: "{user_query}"
SQL Query:
"""
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system",
             "content": "You are an expert Text-to-SQL assistant."},
            {"role": "user", "content": prompt}
        ],
        model       = GROQ_MODEL_FOR_SQL,
        temperature = 0.1,
        max_tokens  = 4096
    )
    generated_sql = strip_think_tags(response.choices[0].message.content)
    if generated_sql == "NO_SQL": return None
    if not (generated_sql.upper().startswith("SELECT")
            and generated_sql.rstrip().endswith(';')):
        return None
    return generated_sql

def find_relevant_attributes_with_sql(generated_sql: str):
    if not generated_sql: return []
    sql_to_run = generated_sql.rstrip().rstrip(';')
    try:
        res = supabase.rpc("execute_readonly_sql", {"q": sql_to_run}).execute()
        if not res.data: return []
        first_key = next(iter(res.data[0].keys()))
        return [row if isinstance(row, dict) else ast.literal_eval(row[first_key])
                for row in res.data]
    except Exception as e:
        st.warning(f"SQL error: {e}")
        return []

def format_context(md_chunks, attr_rows):
    context = []
    if md_chunks:
        context.append("Context from Standards Document:")
        for i, ch in enumerate(md_chunks):
            context.append(f"- Doc {i+1} ({ch.get('filename')}): "
                           f"{ch.get('content')[:400]}…")
    if attr_rows:
        context.append("\nContext from Attributes Table:")
        for i, row in enumerate(attr_rows):
            row_preview = ', '.join(f"{k}: {v}" for k,v in row.items() if v)[:350]
            context.append(f"- Row {i+1}: {row_preview}…")
    return "\n".join(context) if context else ""

def get_groq_chat_response(prompt, have_ctx=True):
    system_msg = ("You are a helpful assistant… answer only from context."
                  if have_ctx else
                  "The knowledge base contained nothing relevant; say so.")
    resp = groq_client.chat.completions.create(
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":prompt}],
        model=GROQ_MODEL_FOR_ANSWER,
        temperature=0.1)
    return strip_think_tags(resp.choices[0].message.content)

# ──────────────────────────────────────────────────────
# 4.  STREAMLIT UI
# ──────────────────────────────────────────────────────
st.title("🔧 LEOparts Standards & Attributes Chatbot")

with st.sidebar:
    st.markdown("### Settings")
    show_debug = st.checkbox("Show debug info", False)

leoni_schema = """(id bigint, Number text, Name text, "Object Type Indicator" text,
Context text, Version text, State text, "Last Modified" timestamptz,
Created_on timestamptz, "Sourcing Status" text, "Material Filling" text,
Material_Name text, "Max. Working Temperature [°C]" numeric,
"Min. Working Temperature [°C]" numeric, Colour text, "Contact Systems" text,
Gender text, "Housing Seal" text, "HV Qualified" text, "Length [mm]" numeric,
"Mechanical Coding" text, "Number Of Cavities" numeric,
"Number Of Rows" numeric, "Pre-assembled" text, Sealing text,
"Sealing Class" text, "Terminal Position Assurance" text,
"Type Of Connector" text, "Width [mm]" numeric, "Wire Seal" text,
"Connector Position Assurance" text, "Colour Coding" text, "Set/Kit" text,
"Name Of Closed Cavities" text, "Pull-To-Seat" text, "Height [mm]" numeric,
Classification text)"""

user_question = st.chat_input("Ask me about parts, colours, etc…")
if user_question:
    with st.spinner("Thinking…"):
        # 1. Text-to-SQL
        sql = generate_sql_from_query(user_question, leoni_schema)
        attr_rows = find_relevant_attributes_with_sql(sql) if sql else []

        # 2. Vector search
        query_emb = get_query_embedding(user_question)
        md_chunks = find_relevant_markdown_chunks(query_emb) if query_emb else []

        # 3. Compose context & answer
        context_text = format_context(md_chunks, attr_rows)
        answer = get_groq_chat_response(
            f"Context:\n{context_text}\n\nUser Question: {user_question}\n"
            "Answer using only the context.") if context_text else \
            get_groq_chat_response(user_question, have_ctx=False)

    # Display
    st.chat_message("user").write(user_question)
    st.chat_message("assistant").write(answer)

    if show_debug:
        st.subheader("🔎 Debug")
        st.write("Generated SQL:", sql)
        st.write("SQL rows:", attr_rows)
        st.write("Vector chunks:", md_chunks)
