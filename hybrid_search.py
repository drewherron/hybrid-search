#!/usr/bin/env python3
"""
Subreddit Hybrid Search - A command-line tool for searching Reddit content.

This script provides a powerful hybrid search capability for Reddit data by combining:
1. Fast lexical search (BM25 via Whoosh) to find relevant documents
2. Semantic re-ranking (using SentenceTransformer embeddings) for better results

Usage:
    python hybrid_search.py --subreddit SUBREDDIT_NAME [--query SEARCH_QUERY]

Examples:
    # Interactive mode with Cornell subreddit
    python hybrid_search.py --subreddit Cornell
    
    # Direct query mode (runs once and exits)
    python hybrid_search.py --subreddit askscience --query "black holes"
    
    # Note: 'subreddit-' prefix is automatically added if not provided
    
Available Subreddits:
    - subreddit-askscience    (r/askscience)
    - subreddit-Cornell       (r/Cornell) 
    - subreddit-politics      (r/politics)
    - subreddit-philosophy    (r/philosophy)
    # and many more provided by Convokit

Note: The first run for each subreddit will download the corpus data.
"""
import os
import re
import shutil
import argparse
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional

# Import all required libraries
from convokit import Corpus, download
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh import index
from whoosh.qparser import QueryParser
from sentence_transformers import SentenceTransformer

# Helper functions to handle different convokit API versions
def get_utterance_ids(corpus: Corpus) -> List[str]:
    """
    Gets utterance IDs using the appropriate method for the convokit version.
    
    Args:
        corpus (Corpus): The Convokit corpus.
        
    Returns:
        List[str]: List of utterance IDs.
    """
    if hasattr(corpus, 'get_utterance_ids'):
        return corpus.get_utterance_ids()
    elif hasattr(corpus, 'iter_utterances'):
        return [str(utt.id) for utt in corpus.iter_utterances()]
    else:
        raise RuntimeError("Unsupported convokit version")

def get_utterance(corpus: Corpus, utt_id: str) -> Any:
    """
    Gets an utterance by ID using the appropriate method for the convokit version.
    
    Args:
        corpus (Corpus): The Convokit corpus.
        utt_id (str): The utterance ID.
        
    Returns:
        Any: The utterance object.
    """
    if hasattr(corpus, 'get_utterance'):
        return corpus.get_utterance(utt_id)
    elif hasattr(corpus, 'iter_utterances'):
        # Create a mapping of IDs to utterances
        for utt in corpus.iter_utterances():
            if str(utt.id) == utt_id:
                return utt
        raise KeyError(f"Utterance {utt_id} not found")
    else:
        raise RuntimeError("Unsupported convokit version")

def get_speaker(utterance: Any) -> str:
    """
    Gets the speaker name from an utterance using the appropriate method for the convokit version.
    
    Args:
        utterance (Any): The utterance object.
        
    Returns:
        str: The speaker name.
    """
    if not hasattr(utterance, 'speaker'):
        return "Unknown"
    
    if utterance.speaker is None:
        return "Unknown"
        
    if hasattr(utterance.speaker, 'id'):
        return utterance.speaker.id
    
    return str(utterance.speaker)

def load_subreddit_corpus(subreddit_name: str) -> Corpus:
    """
    Loads a subreddit corpus from Convokit.

    Args:
        subreddit_name (str): The name of the subreddit corpus (e.g., 'subreddit-Cornell').

    Returns:
        Corpus: The loaded Convokit corpus object.
    
    Raises:
        ValueError: If the subreddit name is invalid.
        ConnectionError: If there's a network issue.
        RuntimeError: For other unexpected errors during download.
    """
    print(f"Downloading corpus for {subreddit_name}...")
    
    # Validate the subreddit name format
    if not subreddit_name.startswith("subreddit-"):
        raise ValueError(f"Invalid subreddit name format: {subreddit_name}. Name should start with 'subreddit-'.")
    
    try:
        corpus = download(subreddit_name)
        
        # Verify corpus has essential methods
        try:
            # Get utterance count safely using our version-agnostic helper
            utterance_count = len(get_utterance_ids(corpus))
        except RuntimeError as e:
            raise RuntimeError(f"Downloaded corpus appears corrupted: {str(e)}")
        
        # Check if we got an empty corpus
        if utterance_count == 0:
            raise ValueError(f"Downloaded corpus '{subreddit_name}' contains no utterances. This may indicate an invalid corpus name.")
            
        print(f"Corpus loaded with {utterance_count} utterances.")
        return corpus
        
    except ConnectionError as e:
        print(f"Connection error when downloading corpus: {e}")
        print("\nPlease check your internet connection and try again.")
        raise ConnectionError(f"Network error while downloading '{subreddit_name}'. Please check your internet connection.") from e
        
    except ValueError as e:
        # ValueError could be raised by ConvoKit if the corpus name doesn't exist
        print(f"Error: {e}")
        print("\nAvailable subreddit corpora include:")
        print("  - subreddit-askscience")
        print("  - subreddit-Cornell")
        print("  - subreddit-politics")
        print("  - subreddit-philosophy")
        raise ValueError(f"Could not find corpus '{subreddit_name}'. Please check the name and try again.") from e
        
    except Exception as e:
        print(f"Error downloading corpus: {e}")
        raise RuntimeError(f"Failed to download subreddit corpus '{subreddit_name}'. Ensure the name is correct and your internet connection is working.") from e


def preprocess_corpus(corpus: Corpus) -> Corpus:
    """
    Performs text cleaning or preprocessing on the corpus.

    Args:
        corpus (Corpus): The Convokit corpus to preprocess.

    Returns:
        Corpus: The preprocessed corpus (could be the same corpus modified in place).
    """
    print("Preprocessing corpus...")
    
    # Use our version-agnostic helper
    for utt_id in get_utterance_ids(corpus):
        utt = get_utterance(corpus, utt_id)
        
        # Convert to lowercase
        cleaned_text = utt.text.lower() if utt.text else ""

        # Remove punctuation (for now this keeps alphanumeric and spaces only)
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

        # Assign the cleaned text back to the utterance
        utt.text = cleaned_text

    return corpus


def build_lexical_index(corpus: Corpus) -> Any:
    """
    Builds a lexical index (e.g., BM25) over the corpus data using Whoosh.
    This version creates a temporary on-disk index.

    Args:
        corpus (Corpus): The preprocessed Convokit corpus.

    Returns:
        (Index): A Whoosh Index object for retrieval.
    """
    print("Building lexical index...")
    # Define a schema for Whoosh. StemmingAnalyzer helps with basic normalization.
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        text=TEXT(analyzer=StemmingAnalyzer(), stored=True)  # Store text for display
    )

    # Create a temporary directory to store the index files
    temp_dir = tempfile.mkdtemp(prefix="whoosh_index_")

    # Create index in this directory
    ix = index.create_in(temp_dir, schema)

    # Open a writer to add documents
    writer = ix.writer()

    # Use our version-agnostic helper
    utterance_ids = get_utterance_ids(corpus)
    for utt_id in utterance_ids:
        utt = get_utterance(corpus, utt_id)
        # Use the utterance's id as the document id
        doc_id = str(utt_id)
        doc_text = utt.text or ""

        # Add the document to the index
        writer.add_document(doc_id=doc_id, text=doc_text)

    # Commit the changes
    writer.commit()

    print(f"Lexical index built with {len(utterance_ids)} documents.")
    return ix


def build_semantic_index(corpus: Corpus) -> Dict[str, Any]:
    """
    Builds a semantic embedding index using a transformer-based model (e.g., Sentence-BERT).
    Stores embeddings in memory for now.

    Args:
        corpus (Corpus): The preprocessed Convokit corpus.

    Returns:
        dict: A dictionary with:
            - 'model': The SentenceTransformer model.
            - 'embeddings': A dict mapping doc_id -> embedding vector (np.array).
    """
    print("Building semantic index...")
    # Load a pre-trained sentence-transformers model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Prepare a dictionary to store doc_id -> embeddings
    embeddings_dict = {}

    # Gather all texts and doc_ids using our version-agnostic helper
    doc_ids = []
    texts = []
    
    utterance_ids = get_utterance_ids(corpus)
    for utt_id in utterance_ids:
        utt = get_utterance(corpus, utt_id)
        doc_ids.append(str(utt_id))
        texts.append(utt.text or "")

    # Compute embeddings in batches for efficiency
    all_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # Map doc_ids to corresponding embeddings
    for i, doc_id in enumerate(doc_ids):
        embeddings_dict[doc_id] = all_embeddings[i]

    print(f"Semantic index built with {len(embeddings_dict)} document embeddings.")
    return {
        "model": model,
        "embeddings": embeddings_dict
    }


def retrieve_lexical(lex_index: Any, query: str, top_n: int = 50) -> List[str]:
    """
    Retrieves top-N documents from the lexical index for a given query.

    Args:
        lex_index (Any): The lexical index structure returned by build_lexical_index().
        query (str): The user's search query string.
        top_n (int): Number of top results to retrieve.

    Returns:
        List[str]: A list of document IDs retrieved from the index.
    """
    # Use the QueryParser from whoosh
    parser = QueryParser("text", schema=lex_index.schema)
    
    # Parse the query string into a Query object
    parsed_query = parser.parse(query)
    
    # Search the index
    with lex_index.searcher() as searcher:
        results = searcher.search(parsed_query, limit=top_n)
        
        # Create a list to store document IDs
        doc_ids = []
        
        # Extract document IDs from search results
        for result in results:
            doc_ids.append(result["doc_id"])
            
        return doc_ids


def rerank_semantic(sem_index: Dict[str, Any], candidate_docs: List[str], query: str) -> List[str]:
    """
    Re-ranks candidate documents using semantic similarity.

    Args:
        sem_index (Dict[str, Any]): The semantic index or embedding store returned by build_semantic_index().
        candidate_docs (List[str]): Candidate documents (IDs) from retrieve_lexical().
        query (str): The user's search query string.

    Returns:
        List[str]: A list of document IDs re-ranked by semantic relevance.
    """
    # Get the model
    model = sem_index["model"]
    
    # Get the embeddings dictionary
    embeddings_dict = sem_index["embeddings"]
    
    # Generate embedding for the query
    query_embedding = model.encode(query)
    
    # Calculate similarity scores for each candidate document
    scores = []
    for doc_id in candidate_docs:
        if doc_id in embeddings_dict:
            # Calculate cosine similarity
            doc_embedding = embeddings_dict[doc_id]
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scores.append((doc_id, similarity))
    
    # Sort documents by similarity score in descending order
    ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Return just the document IDs in ranked order
    return [doc_id for doc_id, _ in ranked_docs]


def display_results(reranked_docs: List[str], corpus: Corpus) -> None:
    """
    Displays the final search results to the user with improved formatting.

    Args:
        reranked_docs (List[str]): The final list of document IDs after re-ranking.
        corpus (Corpus): The Convokit corpus containing the documents.
    """
    if not reranked_docs:
        print("No results found.")
        return
    
    # Get terminal width for better formatting
    try:
        import shutil
        terminal_width = shutil.get_terminal_size().columns
    except (ImportError, AttributeError):
        terminal_width = 80
    
    # Use terminal width for the separator
    separator = "=" * terminal_width
    subseparator = "-" * terminal_width
    
    print(f"\n{separator}")
    print(f"SEARCH RESULTS: Found {len(reranked_docs)} matches")
    print(f"{separator}")
    
    # Use our version-agnostic helper functions
    for i, doc_id in enumerate(reranked_docs, 1):
        try:
            # Get the utterance by ID using our helper
            utterance = get_utterance(corpus, doc_id)
            
            # Get speaker information using our helper
            speaker = get_speaker(utterance)
            
            # Get conversation context if available
            conversation_id = ""
            if hasattr(utterance, 'root') and utterance.root is not None:
                conversation_id = f" | Conversation: {utterance.root}"
            
            # Format and clean the text
            text = utterance.text if utterance.text else ""
            
            # Preserve some punctuation for readability (that was removed during preprocessing)
            # We need to access the original text if possible
            if hasattr(utterance, 'meta') and 'original_text' in utterance.meta:
                text = utterance.meta['original_text']
            
            # Truncate text if it's too long
            if len(text) > 500:
                text = text[:497] + "..."
            
            # Format the output with nice coloring if running in a capable terminal
            try:
                # Only attempt colors if this is likely an interactive terminal
                if hasattr(os, 'isatty') and os.isatty(1):
                    # Check if running on Windows or Unix-like
                    if os.name == 'nt':  # Windows
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                    
                    # ANSI color codes
                    BLUE = '\033[34m'
                    GREEN = '\033[32m'
                    RESET = '\033[0m'
                    
                    print(f"{GREEN}Result {i}{RESET}")
                    print(f"{BLUE}User:{RESET} {speaker}{conversation_id}")
                    print(f"\n{text}\n")
                else:
                    # Plain text formatting for non-interactive terminals
                    print(f"Result {i}")
                    print(f"User: {speaker}{conversation_id}")
                    print(f"\n{text}\n")
            except:
                # Fallback to plain text if any issues with color
                print(f"Result {i}")
                print(f"User: {speaker}{conversation_id}")
                print(f"\n{text}\n")
            
            print(f"{subseparator}")
        except Exception as e:
            print(f"Result {i} [Document ID: {doc_id}]")
            print(f"Document not found in corpus: {str(e)}")
            print(f"{subseparator}")
    
    # Print a footer
    print(f"\nEnd of results. Retrieved {len(reranked_docs)} matches.")
    print(f"{separator}\n")


def main():
    """
    Main program function.

    1. Parse arguments.
    2. Load or prompt for subreddit name.
    3. Load the corpus from Convokit.
    4. Preprocess the corpus.
    5. Build both lexical and semantic indices.
    6. If query is provided as argument, run it once and exit.
       Otherwise, prompt user for queries in a loop.
       - Retrieve top candidates lexically.
       - Re-rank them semantically.
       - Display results.
    7. Exit when the user types an exit command or after single query.
    """
    parser = argparse.ArgumentParser(
        description="Subreddit Hybrid Search - Search Reddit content with BM25 and semantic re-ranking",
        epilog="""
Examples:
  python hybrid_search.py --subreddit Cornell
  python hybrid_search.py --subreddit askscience --query "black holes"

Available subreddits include: askscience, Cornell, politics, philosophy, and more.
        """
    )
    
    parser.add_argument(
        "--subreddit", 
        type=str, 
        help="Name of the subreddit corpus to search (e.g., 'Cornell', 'askscience'). The 'subreddit-' prefix is added automatically if not provided."
    )
    
    parser.add_argument(
        "--query", 
        type=str, 
        help="Search query to run (when provided, the program will run once and exit; otherwise it enters interactive mode)."
    )
    
    args = parser.parse_args()

    # 1. Determine the subreddit name. If blank, prompt.
    subreddit_name = args.subreddit
    if not subreddit_name:
        try:
            subreddit_name = input("Enter the subreddit corpus name (e.g., 'Cornell', 'askscience'): ").strip()
            # Add 'subreddit-' prefix if not already present
            if not subreddit_name.startswith("subreddit-"):
                subreddit_name = "subreddit-" + subreddit_name
        except (EOFError, KeyboardInterrupt):
            print("\nInput interrupted. Exiting.")
            return
    elif not subreddit_name.startswith("subreddit-"):
        subreddit_name = "subreddit-" + subreddit_name

    try:
        # 2. Load the corpus
        corpus = load_subreddit_corpus(subreddit_name)

        # 3. Preprocess
        preprocess_corpus(corpus)

        # 4. Build indices
        lex_index = build_lexical_index(corpus)
        sem_index = build_semantic_index(corpus)

        # 5. Check if we have a query argument
        if args.query:
            query = args.query
            print(f"\nRunning search for query: {query}")
            
            # Retrieve lexical results
            candidate_docs = retrieve_lexical(lex_index, query)

            if not candidate_docs:
                print("No lexical matches found for your query. Try a different search term.")
                return
                
            # Re-rank results
            reranked_docs = rerank_semantic(sem_index, candidate_docs, query)

            # Display results
            display_results(reranked_docs, corpus)
            return
        
        # 6. Query loop if no query argument was provided
        print("\n===== Subreddit Search Ready =====")
        print(f"Corpus: {subreddit_name}")
        print(f"Documents: {len(list(corpus.iter_utterances()))}")
        print("==================================")
        
        while True:
            try:
                query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
                if not query:
                    continue
                    
                if query.lower() in ["exit", "quit", "q"]:
                    print("Exiting search...")
                    break

                # Retrieve lexical results
                candidate_docs = retrieve_lexical(lex_index, query)

                if not candidate_docs:
                    print("No lexical matches found for your query. Try a different search term.")
                    continue
                    
                # Re-rank results
                reranked_docs = rerank_semantic(sem_index, candidate_docs, query)

                # Display results
                display_results(reranked_docs, corpus)
            except EOFError:
                print("\nInput stream closed. Exiting.")
                break
            except KeyboardInterrupt:
                print("\nSearch interrupted. Exiting.")
                break
            except Exception as e:
                print(f"Error during search: {e}")
                print("Please try a different query.")
                
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPopular subreddit corpora you can try include:")
        print("  - subreddit-askscience")
        print("  - subreddit-Cornell")
        print("  - subreddit-politics")
        print("  - subreddit-philosophy")
        
    except ConnectionError as e:
        print(f"Connection error: {e}")
        print("\nPlease check your internet connection and try again.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you're having trouble downloading a subreddit corpus, try a different one like:")
        print("  - subreddit-askscience")
        print("  - subreddit-Cornell")
        print("  - subreddit-politics")

if __name__ == "__main__":
    main()