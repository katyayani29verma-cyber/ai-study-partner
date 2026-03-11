import os
import json

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from pypdf import PdfReader


# -------------------------------
# Load environment variables
# -------------------------------

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------------
# FastAPI setup
# -------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Global vector database
# -------------------------------

vectorstore = None


# -------------------------------
# Request models
# -------------------------------

class RequestData(BaseModel):
    topic: str
    transcript: str


class FlashcardRequest(BaseModel):
    text: str


# -------------------------------
# Root endpoint
# -------------------------------

@app.get("/")
def home():
    return {"status": "AI Study Partner running"}


# -------------------------------
# Upload syllabus PDF
# -------------------------------

@app.post("/upload_syllabus")
async def upload_syllabus(file: UploadFile = File(...)):

    global vectorstore

    pdf = PdfReader(file.file)

    text = ""

    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    return {
        "message": "Syllabus uploaded successfully",
        "chunks_indexed": len(chunks)
    }


# -------------------------------
# Analyze explanation
# -------------------------------

@app.post("/analyze")
def analyze(data: RequestData):

    global vectorstore

    if vectorstore is None:
        return {"error": "Upload syllabus first"}

    docs = vectorstore.similarity_search(data.topic, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI Socratic Tutor.

SYLLABUS CONTEXT:
{context}

STUDENT ANSWER:
{data.transcript}

Evaluate the answer and return STRICT JSON:

{{
"score":1-10,
"difficulty_level":"Beginner/Intermediate/Advanced",
"strengths":[...],
"missing_concepts":[...],
"probing_question":"...",
"hint":"...",
"feedback":"..."
}}
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    try:
        result = json.loads(text)
    except:
        return {"error": "Gemini returned invalid JSON", "raw_output": text}

    # -------------------------------
    # Knowledge Gap Map Feature
    # -------------------------------

    result["knowledge_gap_map"] = {
        "understood": result.get("strengths", []),
        "missing": result.get("missing_concepts", [])
    }

    return result
class RequestData(BaseModel):
    topic: str
    transcript: str

# -------------------------------
# Socratic follow-up question
# -------------------------------

@app.post("/socratic_followup")
def socratic_followup(data: RequestData):

    prompt = f"""
You are a Socratic tutor.

Topic: {data.topic}

Student answer:
{data.transcript}

Ask ONE deeper conceptual question to test understanding.

Return JSON:

{{
"next_question":"..."
}}
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    try:
        return json.loads(text)
    except:
        return {"next_question": text}


# -------------------------------
# Flashcard generator
# -------------------------------

@app.post("/generate_flashcards")
def generate_flashcards(data: FlashcardRequest):

    prompt = f"""
Generate study flashcards from the following notes.

Return JSON:

{{
"flashcards":[
{{"question":"...", "answer":"..."}}
]
}}

Notes:
{data.text}
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    try:
        return json.loads(text)
    except:
        return {"raw_output": text}