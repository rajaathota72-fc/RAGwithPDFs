import streamlit as st
from implementation import extract_text_from_pdf, split_text_into_documents, initialize_rag_model, calculate_rag_price, load_vectorstore_sqlite
from langchain_openai import OpenAIEmbeddings
from config import MODEL_CONFIGS

def upload_and_extract_text():
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf("uploaded_file.pdf")
        return text
    return None

def main():
    st.title("RAG Using Vector Store with Embeddings")
    st.markdown("This application retrieves information from a PDF document using RAG with a vector store and embeddings.")

    # Step 1: Upload PDF and Extract Text
    extracted_text = upload_and_extract_text()
    if extracted_text:
        st.markdown("PDF uploaded successfully. You can now ask questions.")

        # Step 2: Implement RAG using Vector Store
        documents = split_text_into_documents(extracted_text)

        db_path = "vectorstore.db"
        # Check if the vector store already exists
        existing_documents = load_vectorstore_sqlite(db_path)

        if existing_documents is None:
            st.markdown("Creating new embeddings...")
            # Initialize common embeddings
            embeddings = OpenAIEmbeddings()
            rag_chain = initialize_rag_model(documents, embeddings, model_name, db_path)
        else:
            st.markdown("Using existing embeddings...")
            embeddings = OpenAIEmbeddings()
            # User selects the model
            model_name = st.selectbox("Select a model", options=list(MODEL_CONFIGS.keys()))
            rag_chain = initialize_rag_model(existing_documents, embeddings, model_name, db_path)

        st.session_state.rag_chain = rag_chain
        st.session_state.model_name = model_name

    # User Input for Question
    user_input = st.text_input("Enter your question about the document:")
    if user_input and 'rag_chain' in st.session_state:
        rag_chain = st.session_state.rag_chain
        model_name = st.session_state.model_name

        # Step 3: Price Calculation and Comparison
        rag_response, rag_cost = calculate_rag_price(rag_chain, user_input, model_name)

        # Step 4: Display Comparison of Outputs and Pricing
        st.markdown("### RAG Response")
        st.write(rag_response)
        st.markdown(f"**Cost:** ${rag_cost:.6f}")

if __name__ == "__main__":
    main()
