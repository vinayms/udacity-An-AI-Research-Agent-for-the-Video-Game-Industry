import os
from pydantic import BaseModel, Field
from lib.vector_store import GameVectorStore
from lib.llm import LLMClient
from tavily import TavilyClient

class EvaluationReport(BaseModel):
    useful: bool = Field(description="True if the retrieved documents are sufficient to answer the query, False otherwise")
    description: str = Field(description="Brief explanation of why the context is useful or not")

    def __getitem__(self, item):
        if item == "status":
            return "SUFFICIENT" if self.useful else "INSUFFICIENT"
        if item == "explanation":
            return self.description
        return getattr(self, item)

    def get(self, item, default=None):
        try:
            return self[item]
        except AttributeError:
            return default

class GameResearchTools:
    def __init__(self):
        self.vector_store = GameVectorStore()
        self.llm_client = LLMClient()
        
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key and tavily_key != "YOUR_KEY" and not tavily_key.startswith("mock"):
            try:
                self.tavily_client = TavilyClient(api_key=tavily_key)
            except Exception as e:
                print(f"[Tooling] Warning: Failed to initialize Tavily client: {e}. Falling back to mock search.")
                self.tavily_client = None
        else:
            self.tavily_client = None

    def retrieve_game(self, query: str) -> dict:
        """
        Tool 1: Queries the ChromaDB vector database for information about video games.
        Returns a structured dictionary of documents and metadata matching the query.
        """
        results = self.vector_store.query_games(query, n_results=3)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else []
        
        formatted_results = []
        for i in range(len(documents)):
            formatted_results.append({
                "document": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "score": distances[i] if i < len(distances) else 1.0
            })
        return {"query": query, "results": formatted_results}

    def evaluate_retrieval(self, query: str, retrieved_context: str) -> EvaluationReport:
        """
        Tool 2: Uses the LLM to assess whether the retrieved context contains sufficient, 
        accurate, and confident information to fully answer the user query.
        Returns an EvaluationReport Pydantic object.
        """
        system_message = (
            "You are an expert gaming research evaluator. Your task is to analyze whether the "
            "provided search results (Retrieved Context) contain sufficient, precise, and confident details "
            "to answer the User Query.\n"
            "If the context contains the specific game, platform, publisher, release year, or active developer details "
            "needed to answer the query, set useful to true. "
            "If the context is missing key facts, has low confidence, or does not address the specific query, set useful to false.\n\n"
            "Provide a detailed description explaining your decision."
        )
        
        user_message = (
            f"User Query: {query}\n\n"
            f"Retrieved Context:\n{retrieved_context}"
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.llm_client.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages,
                response_format=EvaluationReport,
                temperature=0.0
            )
            return response.choices[0].message.parsed
        except Exception as e:
            # Fallback manual completion parsing
            try:
                raw_response = self.llm_client.get_completion(messages, temperature=0.0)
                cleaned_response = raw_response
                if cleaned_response.startswith("```"):
                    lines = cleaned_response.splitlines()
                    if len(lines) > 2:
                        cleaned_response = "\n".join(lines[1:-1])
                import json
                parsed = json.loads(cleaned_response)
                # Map old keys if LLM returned them
                useful_val = parsed.get("useful", parsed.get("status") == "SUFFICIENT")
                desc_val = parsed.get("description", parsed.get("explanation", "Could not parse evaluation."))
                return EvaluationReport(useful=useful_val, description=desc_val)
            except Exception as ex:
                return EvaluationReport(
                    useful=False,
                    description=f"Evaluation error: {str(e)} -> {str(ex)}"
                )

    def game_web_search(self, query: str) -> dict:
        """
        Tool 3: Performs a web search using the Tavily API to find real-time or missing
        information about video games.
        """
        if not self.tavily_client:
            return {"query": query, "results": [], "error": "Tavily API client is not configured."}

        # Real Tavily Search
        try:
            search_response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=3
            )
            results = search_response.get("results", [])
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")
                })
            return {"query": query, "results": formatted_results}
        except Exception as e:
            return {"query": query, "results": [], "error": str(e)}
