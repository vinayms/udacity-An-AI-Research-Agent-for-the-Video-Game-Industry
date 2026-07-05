# starter/lib/vector_store.py

import os
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from lib.llm import LLMClient

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def __call__(self, input: Documents) -> Embeddings:
        return [self.llm_client.get_embedding(doc) for doc in input]

class GameVectorStore:
    def __init__(self, db_path="starter/games_chromadb", collection_name="games_collection"):
        self.llm_client = LLMClient()
        self.embedding_function = OpenAIEmbeddingFunction(self.llm_client)
        
        # Ensure persistent database directory
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize or fetch the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_game(self, doc_id, document, metadata):
        """
        Adds a single game record to the ChromaDB collection.
        """
        self.collection.add(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata]
        )

    def query_games(self, query_text, n_results=3):
        """
        Queries the ChromaDB collection for the top n_results similar to query_text.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
