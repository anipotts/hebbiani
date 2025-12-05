- biggest problem in AI right now is increasing context windows

  - (agentic) RAG stops at just search & retrieval
  - this is a convolution/info theory problem, with the current capabilities of LLM context windows
  - verticialization is paramount is a fallacy ("AI for _industry_" does not work)
  - generalization beats specification every single time

- hebbia partnered with +50% of largest asset managers and legal firms

- ingestion & indexing is FUNDEMENTAL

  - hooking into as many data sources as possible (boring)
  - how you index data
  - hebbia believes you can do a lot of preprocessing (not keyword search, BM25 index, or long embeddings for semantic search, these are all lame and not for understanding query intent ahead of time)
  - hebbia ingest documents, PREPROCESS/PREPOPULATE depending on the doc type and context with a really rich schema and index of each document (basically doing 90% of the work)
    - interesting search problems in this preprocessing/prepopulating pipeline:
      - parsing documents
      - multi modal documents to figure out formatting and structure and images
      - how do you save the work of the preindexing agents
      - how do you run that index, what are you retrieving over
  - the 10% of the work is from some unpredictable info (this war/crisis/event just broke out here, etc.), from which hebbia then on the fly analyzes and accounts for in the preprocessing and prepopulating the schema.

- RAG does not work for most problems now
  - coined in 2020 by facebook researchers (basis of perplexity aka connecting search engine to LLM)
  - over the web you always have an answer
  - over private, offline and unstructured data you dont always have an answer
  - Hebbia moved away from RAG (retrieval augmented generation) to ISD (integrated stateful reasoning)
    - we need an architecture closer to extending the context window not just searching or calling a certain external tool
  - ISD kinda recursively reads subdocuments (leverages tools and context window), using an agent swarm
  - does a rich decomposition, shows work in the matrix, outputs a final synthesized response
- Hebbia has the most accurate reranker that has ever been released (they say 92%), but they do not use it
- spent a year and a half just training embedding algorithms of ColBERT with multi-embedding architectures for single passage, and training rerankers which to this day academia has not beat

- 250b LLM calls / month (may 2025)
- before maximizer/rate limits, could do 1m tokens a minute, now can do 450-500m tokens a minute (this was a systems problem)
- internal system called maximizer (functions like an air traffic controller)

  - has a handshake between license grants and license requests
    - if job has high or low priority, and wants to use gpt 4o, it can make a license request and license grant will say you have to use xyz model, and maximizer uses the information theoretic maximum utilization of any rate limits

- hallucinations often come from rertreiving from the wrong documents

  - no one really cares about hallucinations, what people care about is when these agents fail from not having the right context or reading documents in the wrong order

- value is way more percieved than actual (less concrete than people like to imagine)
