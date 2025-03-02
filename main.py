import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import google.generativeai as genai
from scraping import scrape_articles 

# Load environment variables from .env file
load_dotenv()

gemini_key = os.getenv("Gemini_API_Key")
news_api_key = os.getenv("News_API_Key")
open_ai_key = os.getenv("Open_AI_Key")

# Configure Gemini API
genai.configure(api_key=gemini_key)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"

# Check if the collection exists, if not create a new one
if collection_name not in client.list_collections():
    collection = client.create_collection(collection_name)
else:
    collection = client.get_collection(collection_name)

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Function to generate embeddings using Gemini API
def get_genai_embedding(text):
    """Generate embeddings using Gemini API."""
    response = genai.embed_content(content=text, model='models/embedding-001')
    embedding = response.get('embedding', None)
    if not embedding:
        raise ValueError(f"Failed to generate embedding for the text: {text[:50]}")
    print("==== Generating embeddings... ====")
    return embedding


# Load documents from the directory
directory_path = "./articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")

# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings for chunk ====")
    doc["embedding"] = get_genai_embedding(doc["text"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into ChromaDB ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )


# Function to query documents
def query_documents(question, n_results=2):
    """Query documents from ChromaDB for the most relevant chunks."""
    question_embedding = get_genai_embedding(question)  # Generate embedding for the question
    results = collection.query(
        query_embeddings=[question_embedding],  # Pass the question's embedding
        n_results=n_results
    )
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):

    client = OpenAI(api_key=open_ai_key)

    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


# Example query and response generation
def main():
    """Main function to run the script."""
    print("Scraping new articles...\n")
    scrape_articles(news_api_key)  # Scrape articles using the news API

    print("Loading documents from the local folder...\n")
    documents = load_documents_from_directory("./articles")  # Load the documents from folder

    if documents:
        print(f"\n{len(documents)} documents found.\n")

        # Split documents and generate embeddings for each chunk
        chunked_documents = []
        for doc in documents:
            chunks = split_text(doc["text"])
            for i, chunk in enumerate(chunks):
                chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

        for doc in chunked_documents:
            doc["embedding"] = get_genai_embedding(doc["text"])

        # Upsert to ChromaDB
        for doc in chunked_documents:
            collection.upsert(ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]])

        while True:
            # Ask user for a question
            question = input("Please enter your question (or type 'exit' to quit): ").strip()

            if question.lower() == 'exit':
                print("Exiting the Q&A system.")
                break

            # Search for relevant documents based on the question
            print(f"\nSearching for answer to the question: '{question}'\n")
            relevant_documents = query_documents(question)

            if relevant_documents:
                print("\nTop relevant documents:\n")
                for idx, doc_content in enumerate(relevant_documents, 1):
                    print(f"Document {idx}: {doc_content[:300]}...")  # Preview of the chunk content
                    print("\n")
                
                # Generate a response based on the relevant documents
                answer = generate_response(question, relevant_documents)
                print(f"Answer: {answer}")
            else:
                print("No relevant documents found for your question.\n")
    else:
        print("No articles found.")

if __name__ == "__main__":
    main()
