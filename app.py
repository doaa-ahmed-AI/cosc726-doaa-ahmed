import uuid
import streamlit as st

from src.agent.chat_titles import get_title, list_chats, make_title, save_title  
from src.agent.graph import graph
from src.agent.validation import find_cited_pages
from src.rag.vector_store import add_pdf


st.set_page_config(page_title="StudyU AI Assistant", page_icon="📚")
st.title("AI Study Assistant")


if "thread" not in st.query_params:
    st.query_params["thread"] = str(uuid.uuid4())
thread_id = st.query_params["thread"]


if st.session_state.get("thread_id") != thread_id:
    st.session_state.thread_id = thread_id
    saved = graph.get_state({"configurable": {"thread_id": thread_id}}).values.get("messages", [])
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


question = st.chat_input("Ask a question about your course material")

if not st.session_state.chat and not question:
    st.info("Upload a lecture PDF in the sidebar, then ask a question about it.")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_details(msg.get("route"), msg.get("sources"), msg.get("missing", ""))


if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat.append({"role": "user", "content": question})

    config = {'configurable': {'thread_id': st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke(
                {"messages": [{"role": "user", "content": question}]}, config=config)

        answer = result['messages'][-1].content
        route = result["message_intent"]
        sources = []
        missing = ""

        if route != "Chat":
           
            cited = find_cited_pages(answer)
            pages = {}
            for d in result["retrieved_docs"]:
                pages.setdefault((d["source"], d["page"]), []).append(d["distance"])
            sources = []
            for (source, page), distances in sorted(pages.items(), key=lambda item: min(item[1])):
                used = "used in the answer" if page in cited else "read, not used"
                sources.append(f"{source}, page {page} · {len(distances)} piece(s) · "
                               f"best distance {min(distances):.2f} · {used}")
            missing = result.get("missing", "")
            route = (f"{route} · searches: {result.get('retry_count', 0) + 1}"
                     f" · failed checks: {result.get('failed_checks', 0)}"
                     f" · query: {result.get('search_query', '')}")

        st.markdown(answer)
        show_details(route, sources, missing)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "route": route, "sources": sources, "missing": missing})



    is_new_chat = get_title(thread_id) is None
    save_title(thread_id, get_title(thread_id) or make_title(question, answer))
    if is_new_chat:
        st.rerun() 