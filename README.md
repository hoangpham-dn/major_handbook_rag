# Major Handbook RAG

A retrieval-augmented question-answering app for Fulbright University's History major handbook. It extracts text and tables from PDF handbooks, indexes them with Chroma, and answers questions through an Ollama-hosted language model. The Streamlit interface displays retrieved handbook excerpts and sources.

## Requirements

- Python 3.11 and Conda (Miniconda or Anaconda) for local setup.
- [Ollama](https://ollama.com/) running on the machine that hosts the model.
- The Ollama models used by the application. The Streamlit app currently configures `qwen3-embedding:0.6b` for embeddings and `qwen3.5:4b-mlx` for answers. The CLI in `src/main.py` uses `qwen3.5:2b-mlx` for answers. Check that these tags are available for your Ollama installation, then download the ones you plan to use with `ollama pull <model-tag>`.
- One or more handbook PDFs under `data/major_handbooks/`. The repository includes `HIS_MajorHandbook.pdf`.

The Python dependencies are listed in [`environment.yml`](environment.yml). They include LangChain and its Ollama/Chroma integrations, ChromaDB, sentence-transformers, Transformers, PyTorch, PyMuPDF, Unstructured, Streamlit, and supporting utilities. Create the environment from that file rather than installing packages individually.

## Installation

Clone the repository and enter its directory, then create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate mh-rag
```

Start Ollama if it is not already running, and pull the models configured for the Streamlit app:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:4b-mlx
```

If you also plan to use the CLI, pull its configured answer model:

```bash
ollama pull qwen3.5:2b-mlx
```

Ollama normally listens at `http://127.0.0.1:11434`, which is the default address used by the local app and CLI.

## Usage

### Streamlit app

From the repository root with the Conda environment activated:

```bash
streamlit run src/app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`) and ask a question about the History major. Use the sidebar button to clear the conversation. Retrieved excerpts and their source filenames appear under each answer when relevant handbook content is found.

To use an Ollama server at a different address, set `OLLAMA_URL` before starting the app:

```bash
OLLAMA_URL=http://localhost:11434 streamlit run src/app.py
```

### Command-line question

The CLI builds or opens the vector store and asks one question. It uses the default sample question when `-q` is omitted:

```bash
python src/main.py --question "What are the requirements for the History major?"
```

Short form:

```bash
python src/main.py -q "How many students are there at Fulbright?"
```

The CLI currently connects to Ollama at `http://127.0.0.1:11434` as configured in `src/main.py`.

### Handbook PDFs and index

Place additional PDFs anywhere under `data/major_handbooks/`; the loader searches that directory recursively. The first run creates a persistent Chroma index in `data/vector_store/`. Subsequent runs load the existing index. If you change or add handbook PDFs and need them re-indexed, remove the existing vector store before restarting so it is rebuilt from the current PDFs.

## Docker deployment

Build and start the Streamlit app using the provided helper script:

```bash
./run_docker.sh
```

The script builds the `mh-handbook-rag` image and publishes the app on port `8501`. It points the container at Ollama on the host using `http://host.docker.internal:11434`. Keep Ollama running on the host and ensure the configured models have been pulled there.

The equivalent commands are:

```bash
docker build -t mh-handbook-rag .
docker run --rm -p 8501:8501 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  mh-handbook-rag
```

Then visit `http://localhost:8501`. On Linux, if `host.docker.internal` is not available, add a host gateway mapping and use the same name:

```bash
docker run --rm -p 8501:8501 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  mh-handbook-rag
```

The Dockerfile copies `data/` into the image, including the handbook PDFs and any existing vector store. To keep a rebuilt index outside the container and persist it across container runs, mount the host data directory:

```bash
docker run --rm -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  mh-handbook-rag
```

For Linux, include the `--add-host=host.docker.internal:host-gateway` option shown above. The container uses the Conda environment defined in `environment.yml` and starts Streamlit on `0.0.0.0:8501`.
