"""
RAG-based Chatbot for Multiscreen Addiction Detector
=====================================================
Uses LangChain + Ollama Cloud + ChromaDB for retrieval-augmented generation.
Persistent chat history via SQLite. Handles both dataset-specific
and general knowledge questions.
"""

import os
import glob
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── LangChain Core ──────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"
CHAT_DB = f"sqlite:///{BASE_DIR / 'chat_history.db'}"
DATASET_HASH_FILE = CHROMA_DIR / ".dataset_hash"

DEFAULT_MODEL = "gpt-oss:120b-cloud"
OLLAMA_CLOUD_BASE_URL = "https://ollama.com"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET → DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
def _load_dataset() -> pd.DataFrame:
    """Find and load the teen multiscreen addiction CSV dataset."""
    csv_files = (
        glob.glob(str(BASE_DIR / '*.csv.xls')) +
        glob.glob(str(BASE_DIR / 'data' / '*.csv.xls')) +
        glob.glob(str(BASE_DIR / '*.csv')) +
        glob.glob(str(BASE_DIR / 'data' / '*.csv'))
    )
    if not csv_files:
        raise FileNotFoundError("Dataset CSV not found!")
    return pd.read_csv(csv_files[0])


def _compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a hash of the dataset to detect changes."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


def _build_row_documents(df: pd.DataFrame) -> list[Document]:
    """Convert each dataset row into a rich text document for the vector DB."""
    docs = []
    for idx, row in df.iterrows():
        text = (
            f"Teen Record #{row.get('ID', idx+1)}:\n"
            f"- Name: {row.get('Name', 'Unknown')}, Age: {row['Age']}, "
            f"Gender: {row['Gender']}, Location: {row.get('Location', 'N/A')}, "
            f"School Grade: {row['School_Grade']}\n"
            f"- Daily Phone Usage: {row['Daily_Usage_Hours']} hours/day, "
            f"Weekend Usage: {row['Weekend_Usage_Hours']} hours/day\n"
            f"- Phone Checks Per Day: {row['Phone_Checks_Per_Day']}, "
            f"Apps Used Daily: {row['Apps_Used_Daily']}\n"
            f"- Phone Time Breakdown: Social Media={row['Time_on_Social_Media']}h, "
            f"Gaming={row['Time_on_Gaming']}h, Education={row['Time_on_Education']}h\n"
            f"- Phone Primary Purpose: {row['Phone_Usage_Purpose']}\n"
            f"- Screen Time Before Bed: {row['Screen_Time_Before_Bed']} hours\n"
            f"- Laptop Usage: Study={row['Laptop_Study_Hours']}h, "
            f"Gaming/Timepass={row['Laptop_Gaming_TimePass_Hours']}h, "
            f"Before Bed={row['Laptop_Usage_Before_Bed_Hours']}h\n"
            f"- Sleep: {row['Sleep_Hours']} hours/night\n"
            f"- Exercise: {row['Exercise_Hours']} hours/day\n"
            f"- Academic Performance: {row['Academic_Performance']}/100\n"
            f"- Social Interactions: {row['Social_Interactions']}/10, "
            f"Family Communication: {row['Family_Communication']}/10\n"
            f"- Mental Health: Anxiety={row['Anxiety_Level']}/10, "
            f"Depression={row['Depression_Level']}/10, Self-Esteem={row['Self_Esteem']}/10\n"
            f"- Parental Control: {'Active' if row['Parental_Control'] == 1 else 'Inactive'}\n"
            f"- ADDICTION LEVEL: {row['Addiction_Level']}/10 "
            f"({'Severe' if row['Addiction_Level'] >= 8.5 else 'High' if row['Addiction_Level'] >= 6.5 else 'Moderate' if row['Addiction_Level'] >= 4.5 else 'Low' if row['Addiction_Level'] >= 2.5 else 'Minimal'})\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source": "dataset_row",
                "row_id": int(row.get('ID', idx+1)),
                "addiction_level": float(row['Addiction_Level']),
                "age": int(row['Age']),
                "gender": str(row['Gender']),
            }
        ))
    return docs


def _build_statistical_documents(df: pd.DataFrame) -> list[Document]:
    """Create summary/statistical documents for richer RAG context."""
    docs = []

    # ── Overall statistics ────────────────────────────────────────────────────
    stats = df.describe()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['ID']
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    overall_text = "DATASET OVERVIEW — Teen Multiscreen Addiction Study\n"
    overall_text += f"Total records: {len(df)} teens surveyed\n"
    overall_text += f"Age range: {int(df['Age'].min())} to {int(df['Age'].max())} years\n"
    overall_text += f"Genders: {', '.join(df['Gender'].unique())}\n"
    overall_text += f"School grades: {', '.join(sorted(df['School_Grade'].unique()))}\n"
    overall_text += f"Phone purposes: {', '.join(df['Phone_Usage_Purpose'].unique())}\n\n"
    overall_text += "KEY STATISTICS:\n"
    for col in numeric_cols:
        overall_text += (
            f"- {col}: mean={df[col].mean():.2f}, "
            f"median={df[col].median():.2f}, "
            f"min={df[col].min():.2f}, max={df[col].max():.2f}, "
            f"std={df[col].std():.2f}\n"
        )
    docs.append(Document(
        page_content=overall_text,
        metadata={"source": "statistics", "type": "overall"}
    ))

    # ── Addiction level distribution ──────────────────────────────────────────
    addiction_text = "ADDICTION LEVEL DISTRIBUTION:\n"
    bins = [(0, 2.5, "Minimal"), (2.5, 4.5, "Low"), (4.5, 6.5, "Moderate"),
            (6.5, 8.5, "High"), (8.5, 10.1, "Severe")]
    for lo, hi, label in bins:
        count = len(df[(df['Addiction_Level'] >= lo) & (df['Addiction_Level'] < hi)])
        pct = count / len(df) * 100
        addiction_text += f"- {label} ({lo}-{hi}): {count} teens ({pct:.1f}%)\n"
    addiction_text += f"\nOverall average addiction level: {df['Addiction_Level'].mean():.2f}/10\n"
    addiction_text += f"Median addiction level: {df['Addiction_Level'].median():.2f}/10\n"
    docs.append(Document(
        page_content=addiction_text,
        metadata={"source": "statistics", "type": "addiction_distribution"}
    ))

    # ── Gender analysis ───────────────────────────────────────────────────────
    gender_text = "ADDICTION BY GENDER:\n"
    for gender, group in df.groupby('Gender'):
        gender_text += (
            f"- {gender}: n={len(group)}, "
            f"avg addiction={group['Addiction_Level'].mean():.2f}, "
            f"avg daily usage={group['Daily_Usage_Hours'].mean():.2f}h, "
            f"avg sleep={group['Sleep_Hours'].mean():.2f}h, "
            f"avg anxiety={group['Anxiety_Level'].mean():.2f}, "
            f"avg depression={group['Depression_Level'].mean():.2f}\n"
        )
    docs.append(Document(
        page_content=gender_text,
        metadata={"source": "statistics", "type": "gender_analysis"}
    ))

    # ── Grade analysis ────────────────────────────────────────────────────────
    grade_text = "ADDICTION BY SCHOOL GRADE:\n"
    for grade, group in df.groupby('School_Grade'):
        grade_text += (
            f"- {grade}: n={len(group)}, "
            f"avg addiction={group['Addiction_Level'].mean():.2f}, "
            f"avg daily usage={group['Daily_Usage_Hours'].mean():.2f}h, "
            f"avg academic performance={group['Academic_Performance'].mean():.1f}\n"
        )
    docs.append(Document(
        page_content=grade_text,
        metadata={"source": "statistics", "type": "grade_analysis"}
    ))

    # ── Correlation insights ──────────────────────────────────────────────────
    corr = df[numeric_cols].corr()['Addiction_Level'].drop('Addiction_Level').sort_values()
    corr_text = "CORRELATION WITH ADDICTION LEVEL (Pearson):\n"
    corr_text += "Strongest POSITIVE correlations (factors that INCREASE addiction):\n"
    for col in corr.nlargest(8).index:
        corr_text += f"  - {col}: r={corr[col]:.3f}\n"
    corr_text += "\nStrongest NEGATIVE correlations (factors that DECREASE addiction):\n"
    for col in corr.nsmallest(8).index:
        corr_text += f"  - {col}: r={corr[col]:.3f}\n"
    docs.append(Document(
        page_content=corr_text,
        metadata={"source": "statistics", "type": "correlations"}
    ))

    # ── Purpose analysis ──────────────────────────────────────────────────────
    purpose_text = "ADDICTION BY PHONE USAGE PURPOSE:\n"
    for purpose, group in df.groupby('Phone_Usage_Purpose'):
        purpose_text += (
            f"- {purpose}: n={len(group)}, "
            f"avg addiction={group['Addiction_Level'].mean():.2f}, "
            f"avg daily usage={group['Daily_Usage_Hours'].mean():.2f}h, "
            f"avg gaming={group['Time_on_Gaming'].mean():.2f}h, "
            f"avg social media={group['Time_on_Social_Media'].mean():.2f}h\n"
        )
    docs.append(Document(
        page_content=purpose_text,
        metadata={"source": "statistics", "type": "purpose_analysis"}
    ))

    # ── Parental control impact ───────────────────────────────────────────────
    pc_text = "IMPACT OF PARENTAL CONTROL:\n"
    for pc, group in df.groupby('Parental_Control'):
        label = "With Parental Control" if pc == 1 else "Without Parental Control"
        pc_text += (
            f"- {label}: n={len(group)}, "
            f"avg addiction={group['Addiction_Level'].mean():.2f}, "
            f"avg daily usage={group['Daily_Usage_Hours'].mean():.2f}h, "
            f"avg sleep={group['Sleep_Hours'].mean():.2f}h, "
            f"avg academic={group['Academic_Performance'].mean():.1f}\n"
        )
    docs.append(Document(
        page_content=pc_text,
        metadata={"source": "statistics", "type": "parental_control"}
    ))

    # ── Sleep & mental health ─────────────────────────────────────────────────
    sleep_text = "SLEEP AND MENTAL HEALTH ANALYSIS:\n"
    sleep_bins = [(3, 5, "Very Low (3-5h)"), (5, 7, "Low (5-7h)"),
                  (7, 8.5, "Normal (7-8.5h)"), (8.5, 11, "Good (8.5h+)")]
    for lo, hi, label in sleep_bins:
        group = df[(df['Sleep_Hours'] >= lo) & (df['Sleep_Hours'] < hi)]
        if len(group) > 0:
            sleep_text += (
                f"- {label}: n={len(group)}, "
                f"avg addiction={group['Addiction_Level'].mean():.2f}, "
                f"avg anxiety={group['Anxiety_Level'].mean():.2f}, "
                f"avg depression={group['Depression_Level'].mean():.2f}, "
                f"avg self-esteem={group['Self_Esteem'].mean():.2f}\n"
            )
    docs.append(Document(
        page_content=sleep_text,
        metadata={"source": "statistics", "type": "sleep_mental_health"}
    ))

    # ── High vs Low addiction comparison ──────────────────────────────────────
    high_addict = df[df['Addiction_Level'] >= 8.5]
    low_addict = df[df['Addiction_Level'] <= 3.0]
    compare_text = "HIGH vs LOW ADDICTION TEENS COMPARISON:\n\n"
    compare_text += f"HIGH addiction (>=8.5/10): {len(high_addict)} teens\n"
    for col in ['Daily_Usage_Hours', 'Sleep_Hours', 'Exercise_Hours',
                'Anxiety_Level', 'Depression_Level', 'Self_Esteem',
                'Academic_Performance', 'Phone_Checks_Per_Day',
                'Screen_Time_Before_Bed', 'Time_on_Social_Media',
                'Time_on_Gaming', 'Laptop_Gaming_TimePass_Hours']:
        if len(high_addict) > 0:
            compare_text += f"  - avg {col}: {high_addict[col].mean():.2f}\n"
    compare_text += f"\nLOW addiction (<=3.0/10): {len(low_addict)} teens\n"
    for col in ['Daily_Usage_Hours', 'Sleep_Hours', 'Exercise_Hours',
                'Anxiety_Level', 'Depression_Level', 'Self_Esteem',
                'Academic_Performance', 'Phone_Checks_Per_Day',
                'Screen_Time_Before_Bed', 'Time_on_Social_Media',
                'Time_on_Gaming', 'Laptop_Gaming_TimePass_Hours']:
        if len(low_addict) > 0:
            compare_text += f"  - avg {col}: {low_addict[col].mean():.2f}\n"
    docs.append(Document(
        page_content=compare_text,
        metadata={"source": "statistics", "type": "high_vs_low_comparison"}
    ))

    # ── Domain knowledge documents ────────────────────────────────────────────
    domain_docs = [
        (
            "ABOUT THIS STUDY AND APPLICATION:\n"
            "This is a Multiscreen Addiction Detector application that uses machine learning "
            "to predict teen screen addiction levels. The application uses 4 ML models: "
            "KNN (K-Nearest Neighbors), SVM (Support Vector Machine), XGBoost, and Random Forest. "
            "The dataset contains 3000 teen records with 28 features covering phone usage, "
            "laptop usage, sleep patterns, mental health indicators, and academic performance. "
            "The ensemble prediction averages all 4 model predictions for a more robust result. "
            "Features include engineered features like Phone_Active_Screen, Phone_Check_Intensity, "
            "Weekend_Weekday_Ratio, Total_Laptop_Hours, Laptop_Productive_Ratio, "
            "Gaming_Cross_Device, Total_All_Screen_Hours, Total_Before_Bed_Screen, "
            "Sleep_Deficit, and Mental_Health_Score.",
            {"source": "domain_knowledge", "type": "about_application"}
        ),
        (
            "SCREEN ADDICTION RISK LEVELS USED IN THIS APPLICATION:\n"
            "- SEVERE ADDICTION (85-100%): Critical level requiring immediate intervention. "
            "Recommended: digital detox, professional help, hard screen limits.\n"
            "- HIGH RISK (65-84%): Significant concern. Recommended: strict app limits, "
            "skill replacement, no phone in bedroom.\n"
            "- MODERATE RISK (45-64%): Requires attention. Recommended: 20-20-20 rule, "
            "notification management, outdoor breaks.\n"
            "- LOW RISK (25-44%): Generally healthy. Recommended: maintain habits, "
            "track weekly, set screen budgets.\n"
            "- MINIMAL RISK (0-24%): Excellent digital wellness. "
            "Recommended: continue current habits, share with peers.",
            {"source": "domain_knowledge", "type": "risk_levels"}
        ),
        (
            "SCIENTIFIC FACTS ABOUT SCREEN ADDICTION IN TEENS:\n"
            "- The American Academy of Pediatrics recommends no more than 2 hours of "
            "recreational screen time per day for teens.\n"
            "- Blue light from screens suppresses melatonin production by up to 22%, "
            "disrupting circadian rhythms.\n"
            "- Teens who use screens >7 hours/day are twice as likely to be diagnosed "
            "with depression or anxiety.\n"
            "- Social media triggers dopamine loops similar to gambling addiction.\n"
            "- Regular exercise for 30+ minutes/day reduces compulsive screen urges by 40%.\n"
            "- Parental monitoring reduces teen screen addiction risk by approximately 30%.\n"
            "- Multitasking across multiple screens reduces attention span and cognitive function.\n"
            "- Screen time before bed reduces sleep quality even if total sleep hours seem adequate.\n"
            "- Gaming addiction is recognized by WHO as a disorder (ICD-11: 6C51).\n"
            "- Teens with higher self-esteem are naturally more resistant to screen addiction.",
            {"source": "domain_knowledge", "type": "scientific_facts"}
        ),
        (
            "STRATEGIES FOR REDUCING SCREEN ADDICTION:\n"
            "1. Digital Detox: Complete screen break for 24-72 hours to reset dopamine pathways.\n"
            "2. App Timers: Use built-in Screen Time (iOS) or Digital Wellbeing (Android).\n"
            "3. Grayscale Mode: Removing color from screens reduces appeal by ~30%.\n"
            "4. No-Phone Zones: Designate bedroom and dining table as phone-free areas.\n"
            "5. Replacement Activities: Replace screen time with physical hobbies, sports, reading.\n"
            "6. Mindfulness & Meditation: 10 mins/day reduces compulsive checking behavior.\n"
            "7. Social Accountability: Share goals with friends/family for 65% better adherence.\n"
            "8. Cognitive Behavioral Therapy (CBT): 80%+ success rate for digital addiction.\n"
            "9. Screen Curfew: No screens 1-2 hours before bed for better sleep.\n"
            "10. Regular Exercise: 30+ mins of physical activity naturally reduces screen cravings.",
            {"source": "domain_knowledge", "type": "reduction_strategies"}
        ),
    ]
    for content, metadata in domain_docs:
        docs.append(Document(page_content=content, metadata=metadata))

    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VECTOR DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
def get_embeddings():
    """Initialize HuggingFace embeddings model (runs locally, no API key)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_db(force_rebuild: bool = False) -> Chroma:
    """
    Build or load the ChromaDB vector store.
    Automatically rebuilds if dataset changes.
    """
    embeddings = get_embeddings()
    df = _load_dataset()
    current_hash = _compute_dataset_hash(df)

    # Check if we need to rebuild
    need_rebuild = force_rebuild or not CHROMA_DIR.exists()
    if not need_rebuild and DATASET_HASH_FILE.exists():
        saved_hash = DATASET_HASH_FILE.read_text().strip()
        if saved_hash != current_hash:
            need_rebuild = True

    if need_rebuild:
        # Build documents
        row_docs = _build_row_documents(df)
        stat_docs = _build_statistical_documents(df)
        all_docs = row_docs + stat_docs

        # Split large docs into smaller chunks for better retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", ", ", " "],
        )
        split_docs = splitter.split_documents(all_docs)

        # Create ChromaDB
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_name="teen_addiction",
        )

        # Save hash
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_HASH_FILE.write_text(current_hash)

        return vectordb
    else:
        # Load existing DB
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name="teen_addiction",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RAG CHAIN WITH PERSISTENT HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

# System prompt — instructs the LLM on its role and context
SYSTEM_PROMPT = """You are an expert AI assistant for the **Multiscreen Addiction Detector** application — a teen digital wellness analysis tool. You are knowledgeable about screen addiction, mental health, teen behavior, and data science.

## Your Capabilities:
1. **Dataset Questions**: Answer questions about the teen multiscreen addiction dataset (3000 records, 28 features). Use the retrieved context to provide accurate, data-driven answers.
2. **General Knowledge**: Answer general questions about screen addiction, digital wellness, mental health, parenting, and technology even if they go beyond the dataset.
3. **Technical Questions**: Explain the ML models used (KNN, SVM, XGBoost, Random Forest), feature engineering, and prediction methodology.
4. **Advice & Recommendations**: Provide evidence-based advice for reducing screen addiction.

## Guidelines:
- When dataset context is provided, ALWAYS use it to ground your answers with specific numbers and statistics.
- If the question is about the dataset and the context contains relevant data, cite specific numbers.
- If the question is general/external, use your knowledge but mention when you're going beyond the dataset.
- Be thorough, detailed, and helpful. Provide structured answers with bullet points when appropriate.
- If asked about a specific teen record, use the retrieved context to provide details.
- Always maintain a professional, supportive, non-judgmental tone especially regarding mental health.
- Remember previous conversation context — the user may refer to earlier parts of the chat.

## Retrieved Context from Dataset & Knowledge Base:
{context}
"""

HUMAN_TEMPLATE = "{question}"


def get_chat_history(session_id: str) -> SQLChatMessageHistory:
    """Get or create a persistent chat history for a session."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=CHAT_DB,
    )


def _create_ollama_llm(
    model_name: str,
    temperature: float,
    ollama_api_key: str | None = None,
):
    """Local Ollama when no API key; Ollama Cloud when OLLAMA_API_KEY is set."""
    api_key = (ollama_api_key or os.getenv("OLLAMA_API_KEY") or "").strip()
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "num_predict": 2048,
        "top_p": 0.9,
    }
    if api_key:
        base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_CLOUD_BASE_URL).rstrip("/")
        kwargs["base_url"] = base_url
        kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {api_key}"},
        }
    return ChatOllama(**kwargs)


def create_rag_chain(
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    ollama_api_key: str | None = None,
):
    """
    Create the full RAG chain with:
    - Vector DB retrieval
    - Chat history awareness
    - Ollama LLM (local or Ollama Cloud via OLLAMA_API_KEY)
    """
    # Vector DB
    vectordb = build_vector_db()
    retriever = vectordb.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diverse results
        search_kwargs={"k": 8, "fetch_k": 20},
    )

    llm = _create_ollama_llm(model_name, temperature, ollama_api_key=ollama_api_key)

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", HUMAN_TEMPLATE),
    ])

    # Helper to format retrieved docs
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Chain: retrieve → format → prompt → llm → parse
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_chat_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history


def ask(
    question: str,
    session_id: str = "default",
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    ollama_api_key: str | None = None,
):
    """
    Ask a question and get a response with full RAG + history.
    Returns the response string.
    """
    chain = create_rag_chain(
        model_name=model_name,
        temperature=temperature,
        ollama_api_key=ollama_api_key,
    )

    response = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )
    return response


def stream_ask(
    question: str,
    session_id: str = "default",
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    ollama_api_key: str | None = None,
):
    """
    Stream a response for real-time token display.
    Yields chunks of the response.
    """
    chain = create_rag_chain(
        model_name=model_name,
        temperature=temperature,
        ollama_api_key=ollama_api_key,
    )

    for chunk in chain.stream(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk


def clear_chat_history(session_id: str = "default"):
    """Clear the chat history for a session."""
    history = get_chat_history(session_id)
    history.clear()


def get_all_sessions() -> list[str]:
    """Get all available chat session IDs."""
    import sqlite3
    db_path = str(BASE_DIR / 'chat_history.db')
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT DISTINCT session_id FROM message_store ORDER BY rowid DESC"
        )
        sessions = [row[0] for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception:
        return []


def get_session_preview(session_id: str) -> str:
    """Get a preview of the first message in a session."""
    history = get_chat_history(session_id)
    messages = history.messages
    if messages:
        first_human = next((m.content for m in messages if m.type == "human"), None)
        if first_human:
            return first_human[:80] + ("..." if len(first_human) > 80 else "")
    return "Empty session"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INITIALIZATION (pre-build vector DB on first import)
# ═══════════════════════════════════════════════════════════════════════════════
def initialize_chatbot():
    """
    Pre-build the vector database if needed.
    Call this on app startup.
    Returns True if successful, False otherwise.
    """
    try:
        build_vector_db()
        return True
    except Exception as e:
        print(f"Chatbot initialization error: {e}")
        return False
