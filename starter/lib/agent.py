import json
import os
from lib.llm import LLMClient
from lib.messages import MessageHistory
from lib.tooling import GameResearchTools

class UdaPlayAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.tools = GameResearchTools()
        self.sessions = {} # maps session_id -> MessageHistory
        self.system_prompt = (
            "You are UdaPlay, an expert AI Research Assistant for the video game industry.\n"
            "Your goal is to provide accurate, comprehensive, and well-structured reports/answers "
            "about video games, including details on developers, release dates, platforms, genres, and publishers.\n\n"
            "You have access to the following tools:\n"
            "1. `retrieve_game`: Search the local vector database for game details. Always call this first.\n"
            "2. `evaluate_retrieval`: Evaluate whether retrieved local database results contain sufficient information to answer the query.\n"
            "3. `game_web_search`: Perform a web search to find missing or up-to-date details.\n\n"
            "Instructions:\n"
            "- Always analyze the conversation history first. If the user's query is a follow-up question (e.g. 'What platform was that on?') and the information is already in the history, answer directly without calling tools.\n"
            "- Otherwise, you MUST call `retrieve_game` first to search the local database for game information.\n"
            "- Once you get the local results, you MUST call `evaluate_retrieval` to evaluate if they are sufficient to answer the question.\n"
            "- If `evaluate_retrieval` reports that local database results are INSUFFICIENT (useful is false), you MUST call `game_web_search` to search the web for the missing details.\n"
            "- If `evaluate_retrieval` reports that results are SUFFICIENT (useful is true), do NOT call `game_web_search`.\n"
            "- Always cite your sources clearly in your final response (e.g. 'Source: Local Database' or using the URL from web results).\n"
            "- Present your final output in a clean, natural, and readable format."
        )
        self.memory_counter = 1000

    def query(self, user_query: str, session_id: str = "default") -> str:
        """
        Processes a single user query through an LLM-driven tool calling loop.
        Manages history on a per-session basis using session_id.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = MessageHistory(system_prompt=self.system_prompt)
        
        history = self.sessions[session_id]
        
        # Add user query to conversation history
        history.add_user_message(user_query)
        
        called_web_search = False
        loop_count = 0
        max_loops = 10
        
        print(f"\n[Agent Workflow] Starting query: '{user_query}' in session: '{session_id}'")
        
        # Define the tools schemas for tool calling
        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_game",
                    "description": "Queries the ChromaDB vector database for information about video games (Name, Platform, YearOfRelease, Description). Always call this first when researching a game.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (e.g. the name of the game)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_retrieval",
                    "description": "Evaluates if the retrieved database results contain sufficient, precise, and confident details to answer the user query. Pass the original query/question and the retrieved results to evaluate.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The original user query or question."
                            },
                            "retrieved_context": {
                                "type": "string",
                                "description": "The formatted text content of the retrieved results from retrieve_game."
                            }
                        },
                        "required": ["query", "retrieved_context"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "game_web_search",
                    "description": "Performs a web search using Tavily to find real-time, missing, or up-to-date video game details. Only call this tool if evaluate_retrieval determines that the retrieved local database results are INSUFFICIENT.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search the web for."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        while loop_count < max_loops:
            loop_count += 1
            messages = history.get_messages()
            
            response = self.llm_client.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                temperature=0.0
            )
            
            message = response.choices[0].message
            tool_calls = message.tool_calls
            
            if not tool_calls:
                final_answer = message.content or ""
                history.add_assistant_message(final_answer)
                
                # Learn from web searches (Long-term memory persistence)
                if called_web_search:
                    print("[Agent Workflow] Persisting web search results in local database memory...")
                    self.persist_to_memory(user_query, final_answer)
                    
                print("[Agent Workflow] Output generated successfully.")
                return final_answer
            
            # Append the assistant message with tool calls to history
            history.messages.append(message)
            
            for tool_call in tool_calls:
                tool_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                print(f"🤖 [Tool Call] Calling tool '{tool_name}' with arguments: {tool_args}")
                
                # Execute tool
                if tool_name == "retrieve_game":
                    res = self.tools.retrieve_game(tool_args["query"])
                    results = res.get("results", [])
                    if not results:
                        tool_result = "No games found in the local database."
                    else:
                        chunks = []
                        for idx, r in enumerate(results, start=1):
                            chunks.append(
                                f"Result {idx} (Source: Local Database):\n"
                                f"Content: {r['document']}\n"
                                f"Metadata: {r['metadata']}\n"
                            )
                        tool_result = "\n".join(chunks)
                        
                elif tool_name == "evaluate_retrieval":
                    q = tool_args.get("query", tool_args.get("question", user_query))
                    ctx = tool_args.get("retrieved_context", tool_args.get("retrieved_docs", ""))
                    if isinstance(ctx, list):
                        ctx = "\n".join([str(item) for item in ctx])
                    res = self.tools.evaluate_retrieval(q, ctx)
                    tool_result = json.dumps({
                        "useful": res.useful,
                        "description": res.description,
                        "status": res["status"],
                        "explanation": res["explanation"]
                    })
                    
                elif tool_name == "game_web_search":
                    called_web_search = True
                    q = tool_args.get("query", tool_args.get("question", user_query))
                    res = self.tools.game_web_search(q)
                    results = res.get("results", [])
                    if not results:
                        tool_result = "No web search results found."
                    else:
                        chunks = []
                        for idx, r in enumerate(results, start=1):
                            chunks.append(
                                f"Web Result {idx} (Title: {r['title']}, Source URL: {r['url']}):\n"
                                f"Content: {r['content']}\n"
                            )
                        tool_result = "\n".join(chunks)
                        
                else:
                    tool_result = f"Error: Unknown tool {tool_name}"
                
                # Print short trace
                short_result = tool_result[:150] + "..." if len(tool_result) > 150 else tool_result
                print(f"🔍 [Tool Result] Tool '{tool_name}' returned: {short_result}")
                
                # Append tool result to history
                history.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": tool_result
                })
        
        return "Agent error: max loop limit exceeded."

    def invoke(self, question: str, session_id: str = "default") -> str:
        """
        Alias for query to match the invoke pattern.
        """
        return self.query(question, session_id=session_id)

    def persist_to_memory(self, query: str, answer: str):
        """
        Parses key findings from the web search final answer and writes them back 
        to ChromaDB, so the agent has internal knowledge for future similar queries.
        """
        try:
            doc_id = f"mem_{self.memory_counter}"
            self.memory_counter += 1
            
            document = f"Learnt Game Data from Research: {answer}"
            metadata = {
                "source": "Agent Web Search Memory",
                "original_query": query
            }
            
            self.tools.vector_store.add_game(doc_id, document, metadata)
            print(f"[Long-Term Memory] Successfully saved new knowledge under ID '{doc_id}'")
        except Exception as e:
            print(f"[Long-Term Memory] Warning: Failed to save knowledge to memory: {e}")
