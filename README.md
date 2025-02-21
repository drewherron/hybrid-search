# Subreddit Hybrid Search

A command-line tool for searching Reddit content using hybrid search technology. This tool combines the speed of lexical search (BM25) with the understanding of semantic search to find the most relevant Reddit posts and comments.

## Features

- **Powerful Hybrid Search**: Combines traditional BM25 keyword search with semantic understanding
- **Interactive Mode**: Search a subreddit with real-time results in a CLI interface
- **Direct Query Mode**: Run a single query from the command line
- **Colorized Output**: Results display formatted for easy reading in the terminal
- **Multiple Subreddits**: Works with any subreddit corpus available in Convokit

## Installation

```bash
# Clone repository (or download code)
git clone https://github.com/yourusername/subreddit-search.git
cd subreddit-search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Interactive mode
python hybrid_search.py --subreddit SUBREDDIT_NAME

# Direct query mode (run once and exit)
python hybrid_search.py --subreddit SUBREDDIT_NAME --query "your search query"
```

### Examples

```bash
# Search r/askscience interactively
python hybrid_search.py --subreddit askscience

# Search r/Cornell for "computer science" and exit
python hybrid_search.py --subreddit Cornell --query "computer science"
```

**Note**: The `subreddit-` prefix is automatically added if not provided.

### Available Subreddits

Some popular subreddit corpora available through Convokit:

- askscience
- Cornell
- politics
- philosophy

And many more...

## How It Works

1. **Corpus Loading**: Downloads and processes a subreddit corpus via Convokit
2. **Lexical Indexing**: Creates a BM25 index using Whoosh for keyword search
3. **Semantic Indexing**: Builds embeddings for each document using SentenceTransformer
4. **Hybrid Search Process**:
   - First finds keyword matches using BM25 lexical search
   - Then re-ranks those matches based on semantic similarity to the query
   - Returns results in order of relevance

## Requirements

- Python 3.7+
- convokit (for corpus management)
- whoosh (for BM25 search)
- sentence-transformers (for semantic search)
- numpy (for numerical operations)
- PyTorch (required by sentence-transformers)

## Troubleshooting

If you encounter issues with PyTorch installation, try installing a specific version compatible with your system:

- CPU only: `pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cpu`
- CUDA 11.6: `pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`

## License

[MIT License](LICENSE)