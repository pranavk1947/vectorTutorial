# Vector Embedding Search with Hugging Face and Neon

A practical tutorial demonstrating how to build a semantic search system for healthcare data using Hugging Face transformers and Neon's PostgreSQL vector database.

## Features

- **Semantic Search**: Find similar patient records based on meaning, not just keywords
- **Healthcare Examples**: Drug interaction detection, temporal patient tracking
- **Vector Storage**: Efficient similarity search using pgvector extension
- **Real-world Use Cases**: Demonstrates practical applications in healthcare data management

## Architecture

- **Text Embeddings**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2` model
- **Vector Database**: Neon PostgreSQL with pgvector extension
- **Similarity Search**: Cosine distance for finding related records

## Prerequisites

- Python 3.8+
- Neon database account (free tier available)
- Git

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd vectorTutorial
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Neon Database

1. Create a free account at [Neon](https://neon.tech)
2. Create a new database
3. Copy your connection string from the Neon dashboard

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
NEON_DB_URL=postgresql://[user]:[password]@[endpoint]/[database]?sslmode=require
```

**Important**: Never commit your `.env` file to version control.

## Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the examples
python main.py
```

### Example Output

```
Query: Patient has headache and feels tired.
Similar records:
Patient: patient_001, Type: chat, Distance: 0.1456
Content: Patient reports mild headache and fatigue.
-----
```

## Code Structure

```
vectorTutorial/
├── main.py          # Main examples and demonstrations
├── embedder.py      # Text embedding functionality
├── db.py           # Database operations and queries  
├── requirements.txt # Python dependencies
├── .env            # Environment variables (create this)
└── README.md       # This file
```

## Key Components

### TextEmbedder (`embedder.py`)
Converts text into high-dimensional vectors using Hugging Face transformers.

### Database Operations (`db.py`)
Handles PostgreSQL connections and vector similarity queries.

### Examples (`main.py`)
Three practical demonstrations:
1. **Basic Patient Search** - Find similar symptoms
2. **Drug Interaction Detection** - Identify related medications
3. **Temporal Patient Tracking** - Track patient progress over time

## How Vector Search Works

1. **Text → Vector**: Convert patient records into numerical embeddings
2. **Storage**: Store vectors in PostgreSQL with pgvector extension
3. **Query**: Convert search terms into vectors
4. **Similarity**: Find closest vectors using cosine distance
5. **Results**: Return most similar patient records

## Use Cases

- **Clinical Decision Support**: Find similar cases for treatment guidance
- **Drug Safety**: Identify potential medication interactions
- **Research**: Discover patterns in patient data
- **Documentation**: Improve medical record search and retrieval

## Extending the Code

### Add New Data Types
```python
# In main.py, add new record types
new_records = [
    ("patient_id", "lab_result", "Blood glucose: 120 mg/dL"),
    ("patient_id", "imaging", "CT scan shows normal findings")
]
```

### Custom Queries
```python
# Search for specific conditions
query_embedding = embedder.embed("diabetes treatment response")
results = query_similar_vectors(query_embedding, top_k=10)
```

## Troubleshooting

### Common Issues

**1. Module Not Found Error**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**2. Database Connection Error**
- Check your `.env` file has the correct NEON_DB_URL
- Verify your Neon database is running
- Ensure your IP is whitelisted in Neon dashboard

**3. pgvector Extension Error**
The code automatically enables the pgvector extension, but if you see errors:
- Ensure your Neon database supports pgvector
- Check database permissions

### Performance Tips

- Use appropriate `top_k` values for queries (default: 5)
- Consider batch insertions for large datasets
- Monitor embedding model memory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Neon Documentation](https://neon.tech/docs)
- [pgvector Extension](https://github.com/pgvector/pgvector)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
