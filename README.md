# AI-Powered RAG Q&A System with Embeddings
This project is designed to scrape articles, process them, and provide a question-answering system using embeddings and a vector database (ChromaDB). Below is a detailed explanation of the project structure, dependencies, and how to set it up.

---

## Project Structure

The project consists of the following files and folders:

1. **Main Script (`main.py`)**:
   - The main script handles the scraping of articles, loading documents, generating embeddings, and querying the ChromaDB for relevant information.
   - It also integrates with OpenAI and Gemini APIs for generating responses and embeddings.

2. **Scraping Script (`scraping.py`)**:
   - This script is responsible for scraping articles from the web using the NewsAPI and saving them to the `articles` folder.

3. **`.env` File**:
   - This file contains the API keys required for the project:
     - `Gemini_API_Key`: API key for the Gemini API.
     - `News_API_Key`: API key for the NewsAPI.
     - `Open_AI_Key`: API key for OpenAI.

4. **`articles` Folder**:
   - This folder is created by the script to store the scraped articles in `.txt` files. Each file contains the title, description, URL, and content of the article.

5. **`chroma_persistent_storage` Folder**:
   - This folder is created by the ChromaDB client to store the vector database persistently. It contains the embeddings and metadata for the documents.

6. **`__pycache__` Folder**:
   - This folder is automatically generated by Python to store compiled bytecode of the modules. It helps in speeding up the loading of modules when the script is run again.

---

## Dependencies

To run this project, you need the following Python libraries:

- `os`: For interacting with the operating system (e.g., reading files, managing directories).
- `dotenv`: For loading environment variables from the `.env` file.
- `openai`: For interacting with the OpenAI API.
- `chromadb`: For managing the vector database (ChromaDB).
- `google.generativeai`: For interacting with the Gemini API.
- `requests`: For making HTTP requests to the NewsAPI and fetching article content.
- `bs4` (BeautifulSoup): For parsing HTML content of the articles.
- `re`: For cleaning up text using regular expressions.

You can install the required libraries using the following command:

```bash
pip install python-dotenv openai chromadb numpy google-generativeai requests beautifulsoup4
```

## How to Run the Project

#### Set Up Environment Variables

1. Create a `.env` file in the root directory of the project.
2. Add the following keys to the `.env` file:

    ```ini
    Gemini_API_Key=your_gemini_api_key
    News_API_Key=your_news_api_key
    Open_AI_Key=your_openai_api_key
    ```

3. Replace `your_gemini_api_key`, `your_news_api_key`, and `your_openai_api_key` with your actual API keys.

---

#### Set Up the Main Script and Scraping Script

##### `main.py`

##### `load_documents_from_directory(directory_path)`
- **Description**: Loads `.txt` documents from a specified directory.
- **Parameters**:
  - `directory_path`: The path to the directory containing the `.txt` files.
- **Returns**: A list of dictionaries, where each dictionary contains the `id` (filename) and `text` (content) of the document.

##### `split_text(text, chunk_size=1000, chunk_overlap=20)`
- **Description**: Splits documents into chunks for easier processing.
- **Parameters**:
  - `text`: The text to be split into chunks.
  - `chunk_size`: The size of each chunk (default: 1000 characters).
  - `chunk_overlap`: The number of overlapping characters between chunks (default: 20 characters).
- **Returns**: A list of text chunks.

##### `get_genai_embedding(text)`
- **Description**: Uses the Gemini API to generate embeddings for document chunks or questions.
- **Parameters**:
  - `text`: The text for which to generate the embedding.
- **Returns**: The embedding vector as a list of floats.

##### `query_documents(question, n_results=2)`
- **Description**: Generates an embedding for the user's question and queries ChromaDB for relevant document chunks.
- **Parameters**:
  - `question`: The user's question.
  - `n_results`: The number of relevant document chunks to retrieve (default: 2).
- **Returns**: A list of relevant document chunks.

##### `generate_response(question, relevant_chunks)`
- **Description**: Generates a response using OpenAI's GPT-3.5 model based on the retrieved document chunks.
- **Parameters**:
  - `question`: The user's question.
  - `relevant_chunks`: The relevant document chunks retrieved from ChromaDB.
- **Returns**: A generated response as a string.

##### `main()`
- **Description**: Orchestrates the entire process: scrapes articles, processes documents, generates embeddings, and handles user queries.
- **Steps**:
  1. Scrapes articles using the NewsAPI.
  2. Loads documents from the `articles` directory.
  3. Splits documents into chunks and generates embeddings.
  4. Stores embeddings in ChromaDB.
  5. Allows the user to query the system and generates responses using OpenAI's GPT-3.5 model.

---

##### `scraping.py`

##### `scrape_articles(api_key)`
- **Description**: Scrapes articles from NewsAPI based on the "finance" query and saves them as `.txt` files in the `articles` directory.
- **Parameters**:
  - `api_key`: The API key for accessing NewsAPI.
- **Steps**:
  1. Fetches articles from NewsAPI.
  2. Parses the article content using BeautifulSoup.
  3. Saves each article as a `.txt` file in the `articles` directory. Each file contains the title, description, URL, and content of the article.

---

## Key Concepts

### **Embeddings witn Gemini API**
- A vector representation of text that captures its semantic meaning.
- Generated using the Gemini API.
- Used to compare and retrieve relevant document chunks based on similarity.

### **ChromaDB**
- A vector database used to store and query embeddings.
- The chroma_persistent_storage folder is created by the ChromaDB client to store the vector database. 
- Do not delete this folder, as it contains the embeddings and metadata required for querying documents.
- Enables fast and efficient retrieval of relevant document chunks based on semantic similarity.

### **OpenAI GPT**
- Used to generate answers based on the context from retrieved document chunks.
- Leverages the GPT-3.5 model to provide concise and accurate responses to user queries.

### Interact with the System
After the `main.py` script runs, you can enter questions in the terminal.
The system will retrieve the most relevant document chunks and generate a response using OpenAI's GPT-3.5-turbo model. Type exit to quit the system.

### Note
`__pycache__`: This folder is automatically created by Python to store compiled bytecode. It is not part of the project logic and can be ignored. If you want to avoid generating this folder, you can run Python with the -B flag:

`python -B main.py`

### Example Workflow

- The script scrapes articles related to finance using the NewsAPI.
- The articles are saved in the articles folder.
- The script loads the articles, splits them into chunks, and generates embeddings using the Gemini API.
- The embeddings are stored in ChromaDB.
- You can query the system by entering questions in the terminal.
- The system retrieves the most relevant document chunks and generates a response using OpenAI's GPT-3.5-turbo model.

### Troubleshooting

***API Key Errors***

Ensure that the API keys in the .env file are correct and have not expired.
If you encounter errors related to API keys, double-check the .env file and the corresponding API documentation.

***ChromaDB Errors***

If the ChromaDB collection is not created or accessed correctly, ensure that the chroma_persistent_storage folder exists and is writable.

***Scraping Errors***

If the script fails to scrape articles, check the NewsAPI key and ensure that the API is functioning correctly.

### Future Improvements

- Support for More Data Sources: Extend the scraping functionality to include more data sources (e.g., RSS feeds, other news APIs).
- Advanced Querying: Implement more advanced querying techniques, such as filtering by date, source, or relevance score.
- User Interface: Develop a web-based or GUI-based interface for easier interaction with the system.
-Performance Optimization: Optimize the embedding generation and querying process for faster response times.

## Conclusion
This project provides a robust system for scraping, processing, and querying articles using state-of-the-art AI models and a vector database. It is highly customizable and can be extended to support additional features and use cases.
