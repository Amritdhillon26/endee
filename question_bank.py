from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

INDEX_NAME = "interview_questions"
EMBED_DIM  = 384

QUESTIONS = [
    {"text": "Tell me about yourself.", "role": "General", "category": "Behavioral"},
    {"text": "What are your strengths and weaknesses?", "role": "General", "category": "Behavioral"},
    {"text": "Explain supervised vs unsupervised learning.", "role": "Data Scientist", "category": "ML"},
    {"text": "What is overfitting and how do you prevent it?", "role": "Data Scientist", "category": "ML"},
    {"text": "Explain gradient descent.", "role": "ML Engineer", "category": "Deep Learning"},
    {"text": "Explain SOLID principles.", "role": "Software Engineer", "category": "OOP"},
    {"text": "What is REST API?", "role": "Software Engineer", "category": "API"},
    {"text": "Explain Big O notation.", "role": "Software Engineer", "category": "DSA"},
    {"text": "What is Docker?", "role": "DevOps Engineer", "category": "Docker"},
    {"text": "Explain CI/CD pipelines.", "role": "DevOps Engineer", "category": "CI/CD"},
]


class QuestionBank:
    def __init__(self):
        self._model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._client = None
        self._index  = None

    def set_config(self, endee_url: str):
        client = Endee()
        client.set_base_url(f"{endee_url}/api/v1")
        self._client = client
        try:
            existing = [idx.name for idx in client.list_indexes()]
        except Exception:
            existing = []
        try:
            if INDEX_NAME not in existing:
                client.create_index(
                    name=INDEX_NAME,
                    dimension=EMBED_DIM,
                    space_type="cosine",
                    precision=Precision.INT8,
                )
        except Exception:
            pass
        self._index = client.get_index(name=INDEX_NAME)

    def index_all(self):
        vectors = []
        for i, q in enumerate(QUESTIONS):
            vec = self._model.encode(q["text"]).tolist()
            vectors.append({
                "id":     f"q_{i:04d}",
                "vector": vec,
                "meta":   {"text": q["text"], "role": q["role"], "category": q["category"]},
            })
        batch = 50
        for i in range(0, len(vectors), batch):
            self._index.upsert(vectors[i:i+batch])

    def search(self, query: str, top_k: int = 10):
        vec = self._model.encode(query).tolist()
        results = self._index.query(vector=vec, top_k=top_k)
        output = []
        for r in results:
            meta = r.get("meta", {}) if isinstance(r, dict) else (r.meta or {})
            sim  = r.get("similarity", 0) if isinstance(r, dict) else r.similarity
            output.append({
                "text":     meta.get("text", ""),
                "role":     meta.get("role", ""),
                "category": meta.get("category", ""),
                "score":    round(sim, 4),
            })
        return output
