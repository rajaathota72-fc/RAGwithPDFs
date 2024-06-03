# RAG with PDFs

This project demonstrates how to use Retrieval-Augmented Generation (RAG) with a vector store and embeddings to retrieve information from a PDF document. The application is built using Streamlit and integrates with OpenAI's GPT models for question answering.

## Features

- Upload and extract text from PDF files.
- Use Retrieval-Augmented Generation (RAG) to answer questions based on the content of the PDF.
- Store and reuse embeddings using SQLite to avoid computation again.
- Select between different OpenAI models for generating responses.
- Calculate and display the cost of using the OpenAI API for generating responses.


## Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/rajaathota72-fc/rag-with-pdf.git
   cd rag-with-pdf
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project directory and add your OpenAI API key:

   ```env
   OPENAI_API_KEY= sk-proj-xxxxxxxxxxx
   ```

5. Run the Streamlit application:

   ```sh
   streamlit run main.py
   ```

## Project Structure

```
RAG with PDFs/
│
├── .env                      # Environment variables
├── config.py                 # Configuration file for models
├── implementation.py         # Core functionalities for RAG and vector store
├── main.py                   # Streamlit app for UI
├── requirements.txt          # Project dependencies
└── vectorstore.db            # SQLite database for storing embeddings (created after first run)
```

## Usage

1. **Upload PDF and extract text**: Start the Streamlit app and upload a PDF file. The text content of the PDF will be extracted and displayed.
2. **Select model**: Choose an OpenAI model for generating responses.
3. **Ask questions**: Enter your question about the document. The app will use RAG to generate a response based on the content of the PDF.
4. **View response and cost**: The generated response and the cost of using the OpenAI API will be displayed.


## Notes

- The app reuses embeddings if they already exist in the SQLite database to save computation time and costs.
- Ensure you have your OpenAI API key in the `.env` file to use the OpenAI models.
