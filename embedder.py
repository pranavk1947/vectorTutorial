from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class TextEmbedder:
    """
    Text embedding class using Hugging Face transformers.
    
    This class converts text into high-dimensional vector representations (embeddings)
    that capture semantic meaning. Similar texts will have similar embeddings.
    
    The default model 'sentence-transformers/all-MiniLM-L6-v2' is optimized for
    semantic similarity tasks and produces 384-dimensional embeddings.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the text embedder with a pre-trained model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use.
                            Default is a sentence transformer optimized for similarity.
        """
        print(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Set model to evaluation mode for inference
        self.model.eval()
        print(f"Model loaded successfully. Embedding dimension: 384")

    def embed(self, text):
        """
        Convert text into a vector embedding.
        
        The process:
        1. Tokenize text into model input format
        2. Pass through transformer model
        3. Apply mean pooling to get sentence-level embedding
        4. Return as numpy array for database storage
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: 384-dimensional embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(384)
            
        # Tokenize text with padding and truncation
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512  # Limit input length for efficiency
        )
        
        # Generate embeddings without computing gradients (inference only)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply mean pooling to get sentence-level representation
        # This averages token embeddings to create a single vector for the entire text
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convert to numpy array and return first (and only) embedding
        return embeddings[0].numpy()
    
    def embed_batch(self, texts):
        """
        Embed multiple texts efficiently in a single batch.
        
        Args:
            texts (list): List of strings to embed
            
        Returns:
            list: List of embedding vectors
        """
        if not texts:
            return []
            
        # Filter out empty texts
        non_empty_texts = [text if text and text.strip() else " " for text in texts]
        
        # Tokenize all texts together
        inputs = self.tokenizer(
            non_empty_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling for each text in the batch
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convert to list of numpy arrays
        return [emb.numpy() for emb in embeddings]