import chromadb
from app.services.document_loader import get_seed_documents


class ChromaStore:
    def __init__(self, path: str = "./chroma_data", collection_name: str = "fuzzyDL_copilot"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def seed_if_empty(self) -> None:
        if self.collection.count() > 0:
            return

        docs = get_seed_documents()

        self.collection.add(
            ids=[doc["id"] for doc in docs],
            documents=[doc["text"] for doc in docs],
            metadatas=[doc["metadata"] for doc in docs],
        )

    def reset_and_reseed(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.seed_if_empty()

    def query(self, question: str, k: int = 3) -> list[dict]:
        results = self.collection.query(
            query_texts=[question],
            n_results=k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        output = []
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            output.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                }
            )

        return output