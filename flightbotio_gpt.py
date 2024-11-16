# -*- coding: utf-8 -*-
"""FlightBot Demo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ox0gOU8LV0j4YmaaqqtmRGK4UFhVSObo
"""

!pip install langchain openai langchain-community langchain-openai
import pandas as pd
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings, OpenAI  # Import OpenAI model for querying
from langchain.docstore.document import Document
from google.colab import files

def load_csv_data(csv_files, chunk_size=1000):
    """
    Load and combine content from a list of CSV files into Document format.
    Splits large data into chunks if needed.

    Args:
        csv_files (list): List of CSV file paths.
        chunk_size (int): Number of rows per chunk to handle large data.

    Returns:
        list: List of Document objects with content from all CSV files.
    """
    documents = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Split large DataFrame into smaller chunks for better processing
        for start in range(0, df.shape[0], chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            content = chunk.to_string(index=False)
            documents.append(Document(page_content=content, metadata={"source": csv_file, "chunk": start}))
    return documents

def csv_qa(csv_files):
    """
    Interactive question-answering on the content of CSV files.
    Allows multiple queries until user types 'quit'.

    Args:
        csv_files (list): A list of CSV file paths.

    Returns:
        None
    """
    openai_api_key = 
    # Set up OpenAI embeddings for the vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Initialize the OpenAI model for querying the vector store
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    # Load the CSV data into documents
    documents = load_csv_data(csv_files)

    # Create an index with the loaded documents
    index = VectorstoreIndexCreator(embedding=embeddings).from_documents(documents)

    # Interactive loop for multiple queries
    while True:
        # Prompt the user for a query
        query = input("\nEnter your question (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            print("Exiting question-answering session.")
            break

        # Use the LLM to query the vector store and get an answer
        ans = index.query(query, llm=llm)

        print(f"Answer: {ans}")

# Upload CSV files
uploaded = files.upload()

# Save uploaded filenames to a list
csv_files = list(uploaded.keys())

# Start the interactive question-answering session
csv_qa(csv_files)