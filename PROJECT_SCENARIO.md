# Project Scenario: UdaPlay

You’ve been hired as an AI Engineer at a gaming analytics company developing an assistant called UdaPlay. Executives, analysts, and gamers want to ask natural language questions like:

- “Who developed FIFA 21?”
- “When was God of War Ragnarok released?”
- “What platform was Pokémon Red launched on?”
- “What is Rockstar Games working on right now?”

Your agent should:
1. Attempt to answer the question from internal knowledge (about a pre-loaded list of companies and games)
2. If the information is not found or confidence is low, search the web
3. Parse and persist the information in long-term memory
4. Generate a clean, structured answer/report

---

## Project Specifications
In this project, you will build an AI Research Agent called **UdaPlay** designed to answer questions about video games. The agent will be capable of:

### 1. Answering User Questions
Provide details about video games, including:
- Game titles and their details
- Release dates and platforms
- Game descriptions and genres
- Publisher information

### 2. Two-Tier Information Retrieval System
- **Primary**: RAG (Retrieval Augmented Generation) over a local dataset of games
- **Secondary**: Web search using the Tavily API when internal knowledge is insufficient

### 3. Robust Evaluation System
- Assessing the quality of retrieved information
- Determining when to fall back to web search
- Providing confidence levels in answers

### 4. Response Generation
Generate clear, well-structured responses that:
- Cite information sources
- Combine information from multiple sources when needed
- Present information in a natural, readable format
