import uuid

from src.agent.graph import graph

if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        user_message = input("Enter message: ")
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]}, config=config)

        print(result["messages"][-1].content)

        if result["message_intent"] != "Chat":
            pages = [
                f"{d['source']} p.{d['page']} ({d['distance']:.2f})" for d in result["retrieved_docs"]]
            print("sources used:", ", ".join(pages))
            print("missing:", result["missing"] or "(nothing)")
            print(
                f"retries: {result['retry_count']} | last search query: {result['search_query']}")
            print(
                f"failed checks: {result['failed_checks']} | problems: {result['problems'] or '(none)'}")
