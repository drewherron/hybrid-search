#!/usr/bin/env python3
"""
Hybrid Lexical + Semantic Search for a Subreddit Corpus.

Usage:
    python search_reddit.py --subreddit subreddit-NameHere
"""
import os
import shutil
import argparse
import tempfile
import numpy as np
from convokit import Corpus, download
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh import index
from sentence_transformers import SentenceTransformer

def load_subreddit_corpus(subreddit_name: str):
    """
    Loads a subreddit corpus from Convokit.

    Args:
        subreddit_name (str): The name of the subreddit corpus (e.g., 'subreddit-Cornell').

    Returns:
        Corpus: The loaded Convokit corpus object.
    """
    pass


def preprocess_corpus(corpus):
    """
    Performs text cleaning or preprocessing on the corpus.

    We could add other steps like stopword removal, lemmatization, or emoji handling.

    Args:
        corpus (Corpus): The Convokit corpus to preprocess.

    Returns:
        Corpus: The preprocessed corpus (could be the same corpus modified in place).
    """
    for utt in corpus.iter_utterances():
        # Convert to lowercase
        cleaned_text = utt.text.lower()

        # Remove punctuation (for now this keeps alphanumeric and spaces only)
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

        # Assign the cleaned text back to the utterance
        utt.text = cleaned_text

    return corpus

def build_lexical_index(corpus):
    """
    Builds a lexical index (e.g., BM25) over the corpus data using Whoosh.
    This version creates a temporary on-disk index.

    Args:
        corpus (Corpus): The preprocessed Convokit corpus.

    Returns:
        (Index): A Whoosh Index object for retrieval.
    """
    # Define a schema for Whoosh. StemmingAnalyzer helps with basic normalization.
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        text=TEXT(analyzer=StemmingAnalyzer(), stored=False)
    )

    # Create a temporary directory to store the index files
    temp_dir = tempfile.mkdtemp(prefix="whoosh_index_")

    # Create index in this directory
    ix = index.create_in(temp_dir, schema)

    # Open a writer to add documents
    writer = ix.writer()

    for utt in corpus.iter_utterances():
        # We might use the utterance's id or some custom unique ID?
        doc_id = str(utt.id)
        doc_text = utt.text or ""

        # Add the document to the index
        writer.add_document(doc_id=doc_id, text=doc_text)

    # Commit the changes
    writer.commit()

    # Return the Whoosh index
    return ix

def build_semantic_index(corpus):
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
    # Load a pre-trained sentence-transformers model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Prepare a dictionary to store doc_id -> embeddings
    embeddings_dict = {}

    # Gather all texts and doc_ids
    doc_ids = []
    texts = []
    for utt in corpus.iter_utterances():
        doc_ids.append(str(utt.id))
        texts.append(utt.text or "")

    # Compute embeddings in batches for efficiency
    all_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # Map doc_ids to corresponding embeddings
    for i, doc_id in enumerate(doc_ids):
        embeddings_dict[doc_id] = all_embeddings[i]

    return {
        "model": model,
        "embeddings": embeddings_dict
    }

def retrieve_lexical(lex_index, query: str):
    """
    Retrieves top-N documents from the lexical index for a given query.

    Args:
        lex_index (Any): The lexical index structure returned by build_lexical_index().
        query (str): The user's search query string.

    Returns:
        List[Any]: A list of candidate documents (IDs or objects) retrieved from the index.
    """
    pass


def rerank_semantic(sem_index, candidate_docs, query: str):
    """
    Re-ranks candidate documents using semantic similarity.

    Args:
        sem_index (Any): The semantic index or embedding store returned by build_semantic_index().
        candidate_docs (List[Any]): Candidate documents (IDs or objects) from retrieve_lexical().
        query (str): The user's search query string.

    Returns:
        List[Any]: A list of documents re-ranked by semantic relevance.
    """
    pass


def display_results(reranked_docs):
    """
    Displays the final search results to the user.

    Args:
        reranked_docs (List[Any]): The final list of documents after re-ranking.

    Returns:
        None
    """
    pass

def main():
    """
    Main program function.

    1. Parse arguments.
    2. Load or prompt for subreddit name.
    3. Load the corpus from Convokit.
    4. Preprocess the corpus.
    5. Build both lexical and semantic indices.
    6. Prompt user for queries in a loop.
       - Retrieve top candidates lexically.
       - Re-rank them semantically.
       - Display results.
    7. Exit when the user types an exit command.
    """
    parser = argparse.ArgumentParser(description="Hybrid Lexical + Semantic Search for a Subreddit Corpus.")
    parser.add_argument("--subreddit", type=str, help="Name of the subreddit corpus (e.g. 'subreddit-Cornell').")
    args = parser.parse_args()

    # 1. Determine the subreddit name. If blank, prompt.
    subreddit_name = args.subreddit
    if not subreddit_name:
        subreddit_name = input("Enter the subreddit corpus name (e.g., 'subreddit-Cornell'): ").strip()

    # 2. Load the corpus
    corpus = load_subreddit_corpus(subreddit_name)

    # 3. Preprocess
    preprocess_corpus(corpus)

    # 4. Build indices
    lex_index = build_lexical_index(corpus)
    sem_index = build_semantic_index(corpus)

    # 5. Query loop
    while True:
        query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting search...")
            break

        # Retrieve lexical results
        candidate_docs = retrieve_lexical(lex_index, query)

        # Re-rank results
        reranked_docs = rerank_semantic(sem_index, candidate_docs, query)

        # Display results
        display_results(reranked_docs)

if __name__ == "__main__":
    main()
