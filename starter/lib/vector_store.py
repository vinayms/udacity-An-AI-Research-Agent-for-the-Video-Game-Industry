# starter/lib/vector_store.py

import os
import chromadb
from chromadb.utils import embedding_functions
from lib.llm import LLMClient

class GameVectorStore:
    def __init__(self, db_path=None, collection_name="udaplay"):
        self.llm_client = LLMClient()
        
        openai_key = os.getenv("OPENAI_API_KEY")
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small",
            api_base="https://openai.vocareum.com/v1/"
        )
        
        # Robust relative path detection based on execution directory
        if db_path is None:
            if os.path.exists("games_chromadb"):
                db_path = "games_chromadb"
            elif os.path.exists("starter/games_chromadb"):
                db_path = "starter/games_chromadb"
            else:
                if os.path.basename(os.getcwd()) == "starter":
                    db_path = "games_chromadb"
                else:
                    db_path = "starter/games_chromadb"
                    
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
