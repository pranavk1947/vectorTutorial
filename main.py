# Vector Embedding Search Demo for Healthcare Data
# This demo shows how to use semantic search with patient records using:
# - Hugging Face transformers for text embeddings
# - Neon PostgreSQL with pgvector for vector storage and similarity search

from embedder import TextEmbedder
from db import create_patient_vectors_table, insert_patient_vector, query_similar_vectors
import numpy as np

def main():
    """
    Basic example: Semantic search for patient records.
    
    This demonstrates how similar symptoms can be found even when 
    different words are used (e.g., "tired" vs "fatigue").
    """
    print("=== BASIC SEMANTIC SEARCH EXAMPLE ===")
    
    # Step 1: Initialize database and embedding model
    create_patient_vectors_table()
    embedder = TextEmbedder()

    # Step 2: Sample patient data with different ways of describing symptoms
    patient_records = [
        ("patient_001", "chat", "Patient reports mild headache and fatigue."),
        ("patient_001", "report", "MRI scan shows no abnormality."),
        ("patient_002", "chat", "Patient complains of chronic back pain."),
    ]

    # Step 3: Convert text to embeddings and store in vector database
    print("Converting patient records to embeddings and storing...")
    for patient_id, data_type, content in patient_records:
        # Generate 384-dimensional embedding that captures semantic meaning
        embedding = embedder.embed(content)
        insert_patient_vector(patient_id, data_type, content, embedding)

    # Step 4: Perform semantic similarity search
    # Notice how "tired" matches "fatigue" even though they're different words
    query_text = "Patient has headache and feels tired."
    print(f"\nSearching for: '{query_text}'")
    
    # Convert query to embedding and find similar vectors in database
    query_embedding = embedder.embed(query_text)
    results = query_similar_vectors(query_embedding)

    print("Most similar patient records:")
    for patient_id, data_type, content, distance in results:
        similarity_score = 1 - distance  # Convert distance to similarity
        print(f"Patient: {patient_id}, Type: {data_type}")
        print(f"Similarity: {similarity_score:.1%}, Distance: {distance:.4f}")
        print(f"Content: {content}")
        print("-----")

def drug_interaction_example():
    """
    Example 2: Drug Interaction Finder using semantic search.
    
    This shows how to find medications with similar effects or interactions,
    useful for identifying potential drug conflicts or alternatives.
    """
    print("\n=== DRUG INTERACTION FINDER EXAMPLE ===")
    
    # Create table if not exists
    create_patient_vectors_table()
    embedder = TextEmbedder()
    
    # Sample medication data - each with unique patient IDs to avoid duplicates
    medications = [
        ("med_001", "medication", "Patient taking warfarin 5mg daily for blood clot prevention", "Dr. Smith", "visit_001"),
        ("med_002", "medication", "Aspirin 81mg prescribed for heart protection and stroke prevention", "Dr. Jones", "visit_002"), 
        ("med_003", "medication", "Ibuprofen 400mg for joint inflammation and pain relief", "Dr. Brown", "visit_003"),
        ("med_004", "medication", "Metformin 1000mg twice daily for diabetes management", "Dr. Wilson", "visit_004"),
        ("med_005", "medication", "Lisinopril 10mg for high blood pressure control", "Dr. Davis", "visit_005"),
        ("med_006", "medication", "Clopidogrel 75mg for antiplatelet therapy after stent placement", "Dr. Smith", "visit_006"),
    ]
    
    # Insert medication data
    print("Inserting medication records...")
    for patient_id, data_type, content, doctor_id, visit_id in medications:
        embedding = embedder.embed(content)
        insert_patient_vector(patient_id, data_type, content, embedding, doctor_id, visit_id)
    
    # Query for blood thinning medications
    print("\n--- Searching for blood thinning medications ---")
    query_text = "blood thinning medication anticoagulant"
    query_embedding = embedder.embed(query_text)
    results = query_similar_vectors(query_embedding, top_k=3)
    
    print(f"Query: '{query_text}'")
    print("Most similar medications:")
    for patient_id, data_type, content, distance in results:
        print(f"Patient: {patient_id}, Distance: {distance:.4f}")
        print(f"Medication: {content}")
        print("-----")

def temporal_patient_tracking_example():
    """
    Example 3: Temporal Patient Tracking using embeddings.
    
    This demonstrates tracking patient progress over time and finding
    similar recovery patterns across different patients and conditions.
    """
    print("\n=== TEMPORAL PATIENT TRACKING EXAMPLE ===")
    
    embedder = TextEmbedder()
    
    # Sample patient timeline with dates
    patient_timeline = [
        ("patient_007", "consultation", "2024-01-15: Initial consultation for chest pain and shortness of breath", "Dr. Garcia", "visit_007_1"),
        ("patient_007", "test_result", "2024-01-20: EKG results show irregular heart rhythm, possible atrial fibrillation", "Dr. Garcia", "visit_007_2"),
        ("patient_007", "follow_up", "2024-02-01: Follow-up visit, patient reports symptoms improving with medication", "Dr. Garcia", "visit_007_3"),
        ("patient_007", "recovery", "2024-02-15: Patient reports complete recovery, heart rhythm normalized", "Dr. Garcia", "visit_007_4"),
        ("patient_008", "consultation", "2024-01-10: Patient presents with severe migraine headaches", "Dr. Lee", "visit_008_1"),
        ("patient_008", "treatment", "2024-01-25: Started preventive migraine medication, tracking symptoms", "Dr. Lee", "visit_008_2"),
        ("patient_008", "follow_up", "2024-02-10: Significant reduction in migraine frequency and intensity", "Dr. Lee", "visit_008_3"),
    ]
    
    # Insert temporal data
    print("Inserting patient timeline records...")
    for patient_id, data_type, content, doctor_id, visit_id in patient_timeline:
        embedding = embedder.embed(content)
        insert_patient_vector(patient_id, data_type, content, embedding, doctor_id, visit_id)
    
    # Query for recovery patterns
    print("\n--- Searching for recovery and improvement patterns ---")
    query_text = "patient recovery symptoms improving getting better"
    query_embedding = embedder.embed(query_text)
    results = query_similar_vectors(query_embedding, top_k=4)
    
    print(f"Query: '{query_text}'")
    print("Similar recovery patterns:")
    for patient_id, data_type, content, distance in results:
        print(f"Patient: {patient_id}, Type: {data_type}, Distance: {distance:.4f}")
        print(f"Record: {content}")
        print("-----")
    
    # Query for specific condition progression
    print("\n--- Searching for heart-related conditions ---")
    query_text = "heart problems cardiac chest pain rhythm"
    query_embedding = embedder.embed(query_text)
    results = query_similar_vectors(query_embedding, top_k=4)
    
    print(f"Query: '{query_text}'")
    print("Heart-related records:")
    for patient_id, data_type, content, distance in results:
        print(f"Patient: {patient_id}, Type: {data_type}, Distance: {distance:.4f}")
        print(f"Record: {content}")
        print("-----")

def clear_existing_data():
    """
    Clear existing data to prevent duplicates when running examples.
    
    In a production system, you'd want more sophisticated data management,
    but for demos this ensures clean results each time.
    """
    from db import get_db_connection
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM patient_data_vectors;")
        conn.commit()
    conn.close()
    print("Cleared existing data to prevent duplicates.\n")

if __name__ == "__main__":
    # Clear existing data to prevent duplicates
    clear_existing_data()
    
    # Run original example
    main()
    
    # Run new examples
    drug_interaction_example()
    temporal_patient_tracking_example()
