# starter/lib/agent.py

import json
from lib.llm import LLMClient
from lib.messages import MessageHistory
from lib.tooling import GameResearchTools

class UdaPlayAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.tools = GameResearchTools()
        self.history = MessageHistory(
            system_prompt=(
                "You are UdaPlay, an expert AI Research Assistant for the video game industry. "
                "Your goal is to provide accurate, comprehensive, and well-structured reports/answers "
                "about video games, including details on developers, release dates, platforms, genres, and publishers.\n"
                "You must cite your sources (e.g., 'Internal Database' or specific URLs from web searches). "
                "Present your final output in a clean, natural, and readable format."
            )
        )
        # Initialize long-term memory counter for unique IDs
        self.memory_counter = 1000

    def query(self, user_query: str) -> str:
        """
        Processes a single user query through a state-machine workflow:
        START -> RETRIEVE -> EVALUATE -> [WEB_SEARCH] -> GENERATE_RESPONSE -> END
        """
        # Add user query to conversation history
        self.history.add_user_message(user_query)
        
        # State machine variables
        state = "START"
        retrieved_context = ""
        web_context = ""
        evaluation_explanation = ""
        used_web_search = False
        
        print(f"\n[Agent Workflow] Starting query: '{user_query}'")
        
        while state != "END":
            if state == "START":
                state = "RETRIEVE"
                
            elif state == "RETRIEVE":
                print("[Agent Workflow] State: RETRIEVE - Querying Vector Store...")
                retrieval_res = self.tools.retrieve_game(user_query)
                results = retrieval_res.get("results", [])
                
                # Format retrieved context
                context_chunks = []
                for idx, r in enumerate(results, start=1):
                    doc = r["document"]
                    meta = r["metadata"]
                    context_chunks.append(
                        f"Result {idx} (Source: Local Database):\n"
                        f"Content: {doc}\n"
                        f"Metadata: {meta}\n"
                    )
                retrieved_context = "\n".join(context_chunks)
                state = "EVALUATE"
                
            elif state == "EVALUATE":
                print("[Agent Workflow] State: EVALUATE - Assessing result sufficiency...")
                eval_res = self.tools.evaluate_retrieval(user_query, retrieved_context)
                status = eval_res.get("status", "INSUFFICIENT")
                explanation = eval_res.get("explanation", "")
                evaluation_explanation = explanation
                
                print(f"[Agent Workflow] Evaluation status: {status}. Explanation: {explanation}")
                
                if status == "SUFFICIENT":
                    state = "GENERATE_RESPONSE"
                else:
                    state = "WEB_SEARCH"
                    
            elif state == "WEB_SEARCH":
                print("[Agent Workflow] State: WEB_SEARCH - Falling back to Tavily Web Search...")
                used_web_search = True
                search_res = self.tools.game_web_search(user_query)
                results = search_res.get("results", [])
                
                # Format web search context
                search_chunks = []
                for idx, r in enumerate(results, start=1):
                    title = r["title"]
                    url = r["url"]
                    content = r["content"]
                    search_chunks.append(
                        f"Web Result {idx} (Title: {title}, Source URL: {url}):\n"
                        f"Content: {content}\n"
                    )
                web_context = "\n".join(search_chunks)
                state = "GENERATE_RESPONSE"
                
            elif state == "GENERATE_RESPONSE":
                print("[Agent Workflow] State: GENERATE_RESPONSE - Compiling final report...")
                # Construct final prompt with context
                prompt_context = ""
                if retrieved_context:
                    prompt_context += f"--- LOCAL DATABASE RESULTS ---\n{retrieved_context}\n\n"
                if web_context:
                    prompt_context += f"--- WEB SEARCH RESULTS ---\n{web_context}\n\n"
                
                prompt_context += f"Evaluation Assessment of Local Knowledge: {evaluation_explanation}\n\n"
                
                system_instruction = (
                    "Using the provided database and web search results, answer the user query accurately. "
                    "Make sure to cite your sources directly inside the text (e.g., 'Source: Local Database' or using the URL from web results). "
                    "If both local database and web search are present, combine them to give a complete picture. "
                    "Structure your answer with clear sections."
                )
                
                # Generate completion using the LLM with conversational context
                current_messages = self.history.get_messages().copy()
                # Insert the context and instructions in the last user message
                current_messages[-1]["content"] = (
                    f"Context:\n{prompt_context}\n\n"
                    f"Instructions: {system_instruction}\n\n"
                    f"Query: {user_query}"
                )
                
                final_answer = self.llm_client.get_completion(current_messages, temperature=0.2)
                self.history.add_assistant_message(final_answer)
                
                # Learn from web searches (Long-term memory persistence)
                if used_web_search and web_context:
                    print("[Agent Workflow] Persisting web search results in local database memory...")
                    self.persist_to_memory(user_query, final_answer)
                    
                state = "END"
        
        print("[Agent Workflow] State: END - Output generated successfully.")
        return final_answer

    def persist_to_memory(self, query: str, answer: str):
        """
        Parses key findings from the web search final answer and writes them back 
        to ChromaDB, so the agent has internal knowledge for future similar queries.
        """
        try:
            # Generate a summary document for embedding
            doc_id = f"mem_{self.memory_counter}"
            self.memory_counter += 1
            
            # Simple document formatting
            document = f"Learnt Game Data from Research: {answer}"
            metadata = {
                "source": "Agent Web Search Memory",
                "original_query": query
            }
            
            # Add to local vector store
            self.tools.vector_store.add_game(doc_id, document, metadata)
            print(f"[Long-Term Memory] Successfully saved new knowledge under ID '{doc_id}'")
        except Exception as e:
            print(f"[Long-Term Memory] Warning: Failed to save knowledge to memory: {e}")
