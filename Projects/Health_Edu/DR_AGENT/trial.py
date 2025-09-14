import os
import io
import fitz  # pymupdf for PDFs
import pytesseract
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
from fastapi import FastAPI, UploadFile, Form
from datetime import datetime
import psycopg2
from minio import Minio
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ==== FastAPI ====
app = FastAPI()

# ==== MinIO client ====
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
bucket_name = "medical-data"
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

# ==== Postgres ====
pg_conn = psycopg2.connect(
    host="postgres",
    dbname="reportsdb",
    user="postgres",
    password="postgres"
)
pg_conn.autocommit = True
cursor = pg_conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    patient_id TEXT,
    doctor_id TEXT,
    patient_name TEXT,
    doctor_name TEXT,
    file_name TEXT,
    extracted_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
""")

# ==== ChromaDB ====
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="/app/chroma_store"
))
collection = chroma_client.get_or_create_collection("medical_reports")

# ==== Sentence Transformer for embeddings ====
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==== Helpers ====
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        # Try direct text extraction with pymupdf
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        text = text.strip()
        if text:  # If text is extracted successfully, return it
            return text
    except:
        pass  # Fallback to OCR if direct extraction fails

    # Fallback to OCR for scanned PDFs
    try:
        images = convert_from_bytes(file_bytes, dpi=300)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, lang="eng+urd") + "\n"
        return text.strip()
    except Exception as e:
        return f"Error during OCR: {str(e)}"

def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return pytesseract.image_to_string(img, lang="eng+urd").strip()
    except Exception as e:
        return f"Error during image OCR: {str(e)}"

def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    return extract_text_from_image(file_bytes)

# ==== API Routes ====
@app.post("/ingest/")
async def ingest_file(
    file: UploadFile,
    patient_id: str = Form(...),
    doctor_id: str = Form(...),
    patient_name: str = Form(None),
    doctor_name: str = Form(None)
):
    file_bytes = await file.read()

    # Store raw file in MinIO
    minio_client.put_object(
        bucket_name,
        file.filename,
        io.BytesIO(file_bytes),
        length=len(file_bytes),
        content_type=file.content_type
    )

    # Extract text
    extracted_text = extract_text(file_bytes, file.filename)

    # Store metadata + text in Postgres
    cursor.execute(
 Juno
    "INSERT INTO reports (patient_id, doctor_id, patient_name, doctor_name, file_name, extracted_text, created_at) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
    (patient_id, doctor_id, patient_name, doctor_name, file.filename, extracted_text, datetime.utcnow())
)
    report_id = cursor.fetchone()[0]

    # Store embedding in Chroma
    if extracted_text and not extracted_text.startswith("Error"):
        embedding = embedder.encode(extracted_text).tolist()
        collection.add(
            documents=[extracted_text],
            metadatas={
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "patient_name": patient_name,
                "doctor_name": doctor_name,
                "file_name": file.filename
            },
            embeddings=[embedding],
            ids=[str(report_id)]
        )

    return {
        "status": "success",
        "report_id": report_id,
        "filename": file.filename,
        "extracted_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
    }

@app.get("/search/")
async def search_reports(query: str):
    results = collection.query(query_texts=[query], n_results=3)
    return {"results": results}