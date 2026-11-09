import uuid

from langgraph.types import Command

from src.agent.graph import graph

if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        user_message = input("Enter message: ")
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]}, config=config)

        while "__interrupt__" in result:
            request = result["__interrupt__"][0].value
            print("\n⏸️  APPROVAL NEEDED:", request["description"])
            print(request["plan"])
            choice = input("approve / edit / reject? ").strip().lower()
            plan = input(
                "Type the EXACT plan text to share: ") if choice == "edit" else request["plan"]
            result = graph.invoke(
                Command(resume={"choice": choice, "plan": plan}), config=config)

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
                      
            print(f"critic rejections: {result['review_count']} | issues: {result['review_issues'] or '(none)'}")
