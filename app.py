import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfStreamError

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")

st.title("📄 RAG Chatbot")
st.write("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")

    try:
        # Read PDF safely
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.error(
                "No readable text was found in this PDF. "
                "The file may be scanned/image-only or corrupted."
            )
            st.stop()

        st.success("PDF loaded successfully!")

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)

        st.write(f"Created **{len(chunks)}** text chunks.")

        # Create embeddings
        with st.spinner("Creating embeddings..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            db = FAISS.from_texts(
                chunks,
                embeddings
            )

        st.success("Vector database created successfully!")

        # Question input
        question = st.text_input(
            "Ask a question about the PDF:"
        )

        if question:

            with st.spinner("Searching document..."):

                docs = db.similarity_search(
                    question,
                    k=3
                )

            st.subheader("Retrieved Context")

            for i, doc in enumerate(docs, start=1):
                with st.expander(f"Chunk {i}"):
                    st.write(doc.page_content)

    except PdfStreamError:
        st.error(
            "This PDF appears to be damaged or incomplete. "
            "Please try another PDF file."
        )

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")