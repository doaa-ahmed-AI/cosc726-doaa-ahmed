import uuid
import streamlit as st

from main import graph
from src.rag.vector_store import add_pdf


st.set_page_config(page_title="StudyU AI Assistant", page_icon="📚")
st.title("AI Study Assistant")


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []


def show_details(route, sources,missing=""):
    if route:
        st.caption(f"Route: {route}")
    if missing:
        st.caption(f"Not found in the material: {missing}")
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
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat = []
        st.rerun()


if not st.session_state.chat:
    st.info("Upload a lecture PDF in the sidebar, then ask a question about it.")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_details(msg.get("route"), msg.get("sources"), msg.get("missing", ""))

question = st.chat_input("Ask a question about your course material")


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
        missing =""

        if route != "Chat":
            sources = [
                f"{d['source']}, page {d['page']} (distance {d['distance']:.2f})" for d in result["retrieved_docs"]]
            missing = result.get("missing", "")
        st.markdown(answer)
        show_details(route, sources, missing)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "route": route, "sources": sources, "missing": missing})
