import streamlit as st
import os

# langchain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain  # corrected import
from langchain_community.vectorstores import FAISS

# Add at the top with other imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



# LOAD ENV VARIABLES
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['google_api_key'] = os.getenv('GOOGLE_API_KEY')

# Custom CSS for centering and styling
st.markdown("""
<style>
.center {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    margin-top: 20px;
}
.stButton>button {
    color: white;
    background-color: #4CAF50; /* Green */
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    border-radius: 5px;
    cursor: pointer;
}
.stButton>button:hover {
    background-color: #45a049;
}
.card {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Center and display images
st.markdown('<div class="center">', unsafe_allow_html=True)
st.image("logo.png", width=200)
# st.image("character_2.webp", width=100)
st.markdown('</div>', unsafe_allow_html=True)

# Remove duplicate/erroneous image code
# st.markdown('<div class="center">', insafe_allow_html=True)  # typo 'insafe'
# st.image

# st.image("logo.png", width=200)
# Title with icon
st.title("📄 ** Document Q&A** 🤖")

# Initialize ChatGroq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# template
prompt = ChatPromptTemplate.from_template(
    """
    Please answer the questions strictly based on the provided context.
    Ensure the response is accurate, concise and directly addresses the question.

    <context>
    {context}
    </context>

    Questions:
    {input}
    """
)

# embeddings function

def vector_embedding():
    if "vectors" not in st.session_state:
        # Assumes GoogleGenerativeAIEmbeddings and PypdfDirectoryLoader are available in your environment
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"]
            )
        st.session_state.loader = PyPDFDirectoryLoader("./ed_pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # 'chinl_size' corrected
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

prompt1 = st.text_input("Enter your questions from any documents")

if st.button("Load database"):
    vector_embedding()
    st.success("Database is ready for queries")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()  # 'vectors' not 'vector'; correct 'as_reteriver' typo
    # 'create_retrival_chain' should be 'create_retrieval_chain'
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    import time
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})  # fixed dict usage
    response_time = time.process_time() - start

    st.markdown("AI response")
    st.success(response["answer"])  # 'Response' -> 'response'
    st.write(f"Response time: {response_time:.2f} seconds")  # fixed string formatting

    with st.expander("document similarity result"):
        st.markdown("Below are the most relevant document chunks: ")
        for i, doc in enumerate(response.get("context", [])):
            st.markdown(
                f"""
                <div class="card">
                    <p>{doc.page_content}</p>
                </div>
                """, unsafe_allow_html=True
            )
