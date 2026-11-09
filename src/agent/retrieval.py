"""Feature: search the course material."""
from src.agent import config
from src.agent.state import StudyState
from src.rag import vector_store


def retrieve(state: StudyState):
  
    new_docs = vector_store.search(state["search_query"], num_of_results=config.NUM_RESULTS)

   
    docs = list(state["retrieved_docs"])
    seen_texts = {d["text"] for d in docs}
    for doc in new_docs:
        if doc["text"] not in seen_texts:
            docs.append(doc)
            seen_texts.add(doc["text"])  
    return {"retrieved_docs": docs}


def format_docs(docs: list[dict]) -> str:
    
    if not docs:
        return "No course material found."
    return "\n\n".join(f"[{d['source']},page {d['page']}]\n{d['text']}" for d in docs)