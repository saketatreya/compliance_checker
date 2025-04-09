# BankRAG Prototype

This project implements a Retrieval-Augmented Generation (RAG) system for checking compliance of internal bank documents against RBI (Reserve Bank of India) regulations.

## Overview

The system works in two main phases:

1.  **Data Ingestion:**
    *   Scrapes regulatory documents (circulars, notifications) from the RBI website.
    *   Preprocesses these documents (HTML and PDF) into structured text chunks with metadata.
    *   Generates vector embeddings for each chunk using a sentence transformer model.
    *   Indexes these embeddings and metadata into a Qdrant vector database.

2.  **Query Pipeline:**
    *   Accepts an internal bank document (TXT, PDF, DOCX) as input.
    *   Processes the document, potentially splitting it into manageable chunks.
    *   For each chunk, retrieves the most relevant RBI regulation excerpts from the Qdrant database based on semantic similarity.
    *   Uses a Large Language Model (LLM), specifically Google Gemini, to analyze the document chunk against the retrieved regulations.
    *   Generates a compliance report highlighting potential issues and citing relevant RBI rules.

## Project Structure

```
BankRAG/
├── data/                      # Storage for scraped and processed data
│   ├── raw/                   # Raw HTML/PDF from RBI website
│   ├── processed/             # Cleaned and structured text (JSONL)
│   └── embeddings/            # Vector embeddings (JSONL - optional intermediate)
├── qdrant_storage/            # Persistent storage for Qdrant data (via Docker volume)
├── src/                       # Source code
│   ├── scraper/               # Web scraping module
│   ├── preprocessor/          # Text processing (HTML, PDF)
│   ├── embedding/             # Vector embedding generation
│   ├── llm/                   # LLM interface (Gemini)
│   ├── pipeline/              # End-to-end pipelines (ingestion, query)
│   └── interface/             # User interfaces (CLI, Web App)
├── tests/                     # Test suite
│   └── test_document.txt      # Sample document for testing
├── config.py                  # Configuration settings (if needed)
├── docker-compose.yml         # Docker Compose file for Qdrant
├── project_plan.md            # Project plan details
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd BankRAG
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Google API Key:**
    *   Obtain a Google API key with access to the Gemini API.
    *   Set it as an environment variable:
        ```bash
        export GOOGLE_API_KEY='your_api_key_here'
        ```
    *   Alternatively, create a `.env` file in the project root and add `GOOGLE_API_KEY='your_api_key_here'` (requires `python-dotenv` installed and code modification to load it).

5.  **Start Qdrant Vector Database:**
    *   Ensure you have Docker installed.
    *   Run Qdrant using Docker Compose:
        ```bash
        docker-compose up -d
        ```
    *   This will start a Qdrant instance and store its data in the `qdrant_storage` directory.

## Usage

### 1. Data Ingestion

Run the ingestion pipeline to scrape, process, and index RBI regulations. This needs to be done initially and periodically to keep the knowledge base updated.

```bash
python src/interface/cli.py ingest
```

Optional arguments for `ingest`:
*   `--doc-types`: Specify document types (e.g., `--doc-types circulars`). Defaults to `circulars` and `notifications`.
*   `--qdrant-host`, `--qdrant-port`, `--collection`, `--vector-size`, `--batch-size`, `--model`: Configure ingestion parameters (see `cli.py` for defaults).

*(Note: The scraper implementation is currently basic and needs enhancement to handle RBI website structure and pagination fully.)*

### 2. Compliance Analysis (CLI)

Analyze a local document using the command-line interface:

```bash
python src/interface/cli.py analyze /path/to/your/document.pdf 
```

Replace `/path/to/your/document.pdf` with the actual path to your TXT, PDF, or DOCX file.

Optional arguments for `analyze`:
*   `--top-k`: Number of relevant regulations to retrieve (default: 5).
*   `--threshold`: Minimum similarity score (default: 0.7).
*   `--qdrant-host`, `--qdrant-port`, `--collection`, `--embedding-model`, `--llm-model`: Configure analysis parameters.

### 3. Compliance Analysis (Web App)

Launch the Streamlit web interface:

```bash
streamlit run src/interface/web_app.py
```

Open your web browser to the URL provided by Streamlit (usually `http://localhost:8501`). Use the sidebar to upload your document and configure analysis parameters.

## Development Notes

*   **Scraper:** The current `rbi_scraper.py` is basic. It needs significant improvement to handle the actual structure, pagination, and potential dynamic loading on the RBI website.
*   **Preprocessing:** `text_processor.py` includes basic logic for HTML and PDF parsing. It attempts to identify structure but may need refinement based on more diverse RBI document formats.
*   **Embeddings:** Uses `sentence-transformers` (specifically `all-mpnet-base-v2` by default).
*   **Retrieval:** Uses Qdrant for vector storage and retrieval.
*   **LLM:** Uses Google Gemini via LangChain.
*   **Error Handling:** Basic error handling is included, but robustness can be improved.

## Future Enhancements

*   Improve scraper robustness and coverage.
*   Enhance preprocessing to better handle tables, complex layouts, and scanned PDFs (OCR).
*   Implement more sophisticated chunking strategies.
*   Evaluate and potentially fine-tune embedding models.
*   Add support for conversational follow-up questions in the UI.
*   Integrate more data sources (FAQs, Master Directions index pages).
*   Improve UI/UX based on user feedback.
*   Add comprehensive unit and integration tests.
*   Implement user authentication for the web app.