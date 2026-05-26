from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_path="models/embeddings/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_path)

    def encode(self, text):
        return self.model.encode(text).tolist()