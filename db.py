import os
import sys
from dotenv import load_dotenv
import psycopg
import numpy as np

load_dotenv()

# Load Neon database connection string from environment variable
NEON_DB_URL = os.getenv("NEON_DB_URL")

if not NEON_DB_URL:
    print("Error: NEON_DB_URL environment variable is not set.")
    print("Please create a .env file with your Neon database connection string.")
    sys.exit(1)

# Connect to Neon Postgres database with error handling
def get_db_connection():
    """Establish database connection with error handling"""
    try:
        return psycopg.connect(NEON_DB_URL)
    except psycopg.OperationalError as e:
        print(f"Database connection failed: {e}")
        print("Please check your NEON_DB_URL in the .env file.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected database error: {e}")
        sys.exit(1)

def create_patient_vectors_table():
    """Create the patient vectors table with pgvector extension"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Create table for patient data vectors (384 dimensions for all-MiniLM-L6-v2)
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
            print("Database table created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def insert_patient_vector(patient_id, data_type, content, embedding, doctor_id=None, visit_id=None):
    """Insert a patient record with its vector embedding into the database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Convert numpy array to list for PostgreSQL vector type
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
            cur.execute("""
                INSERT INTO patient_data_vectors (patient_id, data_type, content, embedding, doctor_id, visit_id)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (patient_id, data_type, content, embedding_list, doctor_id, visit_id))
            conn.commit()
    except Exception as e:
        print(f"Error inserting vector: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def query_similar_vectors(query_embedding, top_k=5):
    """Query for similar vectors using cosine distance
    
    Args:
        query_embedding: The embedding vector to search for
        top_k: Number of most similar results to return
        
    Returns:
        List of tuples: (patient_id, data_type, content, distance)
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Convert numpy array to list for PostgreSQL vector type
            embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
            
            # Use cosine distance operator <=> for similarity search
            cur.execute("""
                SELECT patient_id, data_type, content, embedding <=> %s::vector AS distance
                FROM patient_data_vectors
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_list, embedding_list, top_k))
            results = cur.fetchall()
            return results
    except Exception as e:
        print(f"Error querying vectors: {e}")
        raise
    finally:
        conn.close()
