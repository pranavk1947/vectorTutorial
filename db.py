import os
from dotenv import load_dotenv
import psycopg
import numpy as np

load_dotenv()

# Load Neon database connection string from environment variable
NEON_DB_URL = os.getenv("NEON_DB_URL")

# Connect to Neon Postgres database
def get_db_connection():
    return psycopg.connect(NEON_DB_URL)

def create_patient_vectors_table():
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create table for patient data vectors
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patient_data_vectors (
                id SERIAL PRIMARY KEY,
                patient_id VARCHAR NOT NULL,
                data_type VARCHAR NOT NULL,
                content TEXT,
                embedding vector(384),
                timestamp TIMESTAMPTZ DEFAULT now(),
                doctor_id VARCHAR,
                visit_id VARCHAR
            );
        """)
        conn.commit()
    conn.close()

def insert_patient_vector(patient_id, data_type, content, embedding, doctor_id=None, visit_id=None):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO patient_data_vectors (patient_id, data_type, content, embedding, doctor_id, visit_id)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (patient_id, data_type, content, embedding.tolist(), doctor_id, visit_id))
        conn.commit()
    conn.close()

def query_similar_vectors(query_embedding, top_k=5):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT patient_id, data_type, content, embedding <=> %s::vector AS distance
            FROM patient_data_vectors
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
        results = cur.fetchall()
    conn.close()
    return results
