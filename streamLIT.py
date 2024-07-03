import streamlit as st
from Rag import RAG
import time
def stream_data(string):
    for word in string.split(" "):
        yield word + " "
        time.sleep(0.07)
@st.cache_resource()
def initialize():
    st.write_stream(stream_data("Setting Things Up..."))
    st.write("---")
    return RAG()
rag = initialize()
st.write(stream_data("Ask Away!"))
st.write("---")
prompt = st.chat_input("Ask me Anything!")
if prompt:
    st.write(stream_data(prompt))
    st.write("---")
    st.write(stream_data("Sure! Give me a moment while I gather the information you need!"))
    st.write("---")
    output = rag.generate(prompt)
    for chunk in output:
        if "actions" in chunk:
            for action in chunk["actions"]:
                st.write_stream(stream_data(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`"))
        elif "steps" in chunk:
            for step in chunk["steps"]:
                st.write_stream(stream_data(f"Thinking...."))
        elif "output" in chunk:
            st.write_stream(stream_data(f'Final Output: {chunk["output"]}'))
        else:
            raise ValueError()
        st.write("---")