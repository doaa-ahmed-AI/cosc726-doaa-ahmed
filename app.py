import uuid
import streamlit as st
from langgraph.types import Command

from src.agent.chat_titles import get_title, list_chats, make_title, save_title
from src.agent.graph import graph
from src.agent.validation import find_cited_pages
from src.rag.vector_store import add_pdf


st.set_page_config(page_title="StudyU AI Assistant", page_icon="📚")
st.title("AI Study Assistant")


if "thread" not in st.query_params:
    st.query_params["thread"] = str(uuid.uuid4())
thread_id = st.query_params["thread"]
config = {"configurable": {"thread_id": thread_id}}


if st.session_state.get("thread_id") != thread_id:
    st.session_state.thread_id = thread_id
    saved = graph.get_state(config).values.get("messages", [])

    st.session_state.chat = [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in saved
    ]


def show_details(route, sources, missing=""):
    if route:
        st.caption(f"Route: {route}")
    if missing:
        st.caption(f"Not found in the searched pages: {missing}")
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.write(f"- {s}")


def answer_entry(result) -> dict:
    """Turn the graph's result into one chat entry: the answer + its route, sources and 'missing'."""
    answer = result["messages"][-1].content
    route = result["message_intent"]
    entry = {"role": "assistant", "content": answer,
             "route": route, "sources": [], "missing": ""}

    if route != "Chat":

        cited = find_cited_pages(answer)
        pages = {}
        for d in result["retrieved_docs"]:
            pages.setdefault((d["source"], d["page"]),
                             []).append(d["distance"])
        for (source, page), distances in sorted(pages.items(), key=lambda item: min(item[1])):
            used = "✅ used in the answer" if page in cited else "read, not used"
            entry["sources"].append(f"{source}, page {page} · {len(distances)} piece(s) · "
                                    f"best distance {min(distances):.2f} · {used}")
        entry["missing"] = result.get("missing", "")
        entry["route"] = (f"{route} · searches: {result.get('retry_count', 0) + 1}"
                          f" · failed checks: {result.get('failed_checks', 0)}"
                          f" · critic rejections: {result.get('review_count', 0)}" 
                          f" · query: {result.get('search_query', '')}")
    return entry


def finish_turn(result, question: str):

    entry = answer_entry(result)
    st.session_state.chat.append(entry)
    save_title(thread_id, get_title(thread_id)
               or make_title(question, entry["content"]))
    st.rerun()  


with st.sidebar:
    st.header("Course material")
    files = st.file_uploader("Upload Lecture PDFs",
                             type="pdf", accept_multiple_files=True)

    if st.button("Add PDFs", disabled=not files):
        for file in files:
            with st.spinner(f"Reading {file.name}..."):
                count = add_pdf(file.getvalue(), file.name)
            st.success(f"Added {file.name} ({count} chunks)")

    st.divider()
    if st.button("New chat"):
        st.query_params["thread"] = str(uuid.uuid4())  # a new id in the URL
        st.rerun()

    st.caption("Your chats")
    for chat_id, title in list_chats():
        is_open = chat_id == thread_id
        if st.button(title, key=f"chat-{chat_id}", type="primary" if is_open else "secondary"):
            st.query_params["thread"] = chat_id  # open that chat
            st.rerun()
pending = graph.get_state(config).interrupts
question = st.chat_input(
    "Ask a question about your course material", disabled=bool(pending))

if not st.session_state.chat and not question:
    st.info("Upload a lecture PDF in the sidebar, then ask a question about it.")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_details(msg.get("route"), msg.get(
            "sources"), msg.get("missing", ""))

if pending:
    request = pending[0].value
    decision = None
    with st.chat_message("assistant"):
        st.warning(f"⏸️ Approval needed: {request['description']}")
        edited_plan = st.text_area(
            "The plan (you can edit it before sharing):", request["plan"], height=250)
        col1, col2, col3 = st.columns(3)
        if col1.button("✅ Approve"):
            decision = {"choice": "approve", "plan": request["plan"]}
        if col2.button("✏️ Share my edited version"):
            decision = {"choice": "edit", "plan": edited_plan}
        if col3.button("❌ Reject"):
            decision = {"choice": "reject", "plan": request["plan"]}

    if decision:
        with st.spinner("Continuing..."):
            
            result = graph.invoke(Command(resume=decision), config=config)
        last_question = [m["content"]
                         for m in st.session_state.chat if m["role"] == "user"][-1]
        finish_turn(result, last_question)


if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke(
                {"messages": [{"role": "user", "content": question}]}, config=config)

    if "__interrupt__" in result:
        st.rerun()  

    finish_turn(result, question)
