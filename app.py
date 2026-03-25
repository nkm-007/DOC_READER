import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask Your PDF (Free Version)")

    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    pdf = st.file_uploader("Upload your PDF file", type="pdf")

    if pdf is not None and st.session_state.knowledge_base is None:
        with st.spinner("Processing PDF..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.success("PDF processed successfully!")

    if st.session_state.knowledge_base:
        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            docs = st.session_state.knowledge_base.similarity_search(user_question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            with st.spinner("Generating answer..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Free model on Groq
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on the provided PDF context. If the answer is not in the context, say 'I don't know based on the provided document.'"
                        },
                        {
                            "role": "user",
                            "content": f"Context from PDF:\n{context}\n\nQuestion: {user_question}"
                        }
                    ],
                    temperature=0.5,
                    max_tokens=500
                )
                st.write(response.choices[0].message.content)


if __name__ == "__main__":
    main()
