#!/usr/bin/env python3
"""
Hybrid Lexical + Semantic Search for a Subreddit Corpus.

Usage:
    python hybrid_search.py --subreddit subreddit-NameHere
"""
import os
import re
import shutil
import argparse
import tempfile
import numpy as np
import random

# Mock implementations for missing dependencies
class MockCorpus:
    """Mock Corpus class for testing without Convokit."""
    def __init__(self, name):
        self.name = name
        self.utterances = []
        # Create some test data
        for i in range(100):
            self.utterances.append({
                'id': f"id_{i}",
                'text': f"This is test utterance {i} talking about {'Python' if i % 5 == 0 else 'programming'} and {'search' if i % 3 == 0 else 'data'}"
            })
    
    def iter_utterances(self):
        for utt in self.utterances:
            yield MockUtterance(utt['id'], utt['text'])
            
class MockUtterance:
    """Mock Utterance class for testing."""
    def __init__(self, id_str, text):
        self.id = id_str
        self.text = text
        self.speaker = MockSpeaker(f"user_{random.randint(1, 10)}")
        
class MockSpeaker:
    """Mock Speaker class for testing."""
    def __init__(self, id_str):
        self.id = id_str

class MockSentenceTransformer:
    """Mock SentenceTransformer for testing."""
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Using mock SentenceTransformer with model: {model_name}")
        
    def encode(self, texts, batch_size=32, show_progress_bar=True):
        """Return random embeddings of specified dimension."""
        if isinstance(texts, str):
            return np.random.rand(384)  # Return a single embedding
        return np.random.rand(len(texts), 384)  # Return batch of embeddings

# Try to import actual dependencies, but fall back to mocks
try:
    from convokit import Corpus, download
except ImportError:
    print("Convokit not found, using mock implementation")
    Corpus = MockCorpus
    def download(name):
        print(f"Mock downloading corpus for {name}")
        return MockCorpus(name)

try:
    from whoosh.fields import Schema, TEXT, ID
    from whoosh.analysis import StemmingAnalyzer
    from whoosh import index
    from whoosh.qparser import QueryParser
    WHOOSH_AVAILABLE = True
except ImportError:
    print("Whoosh not found, using mock implementation")
    WHOOSH_AVAILABLE = False
    
    # Mock implementations for Whoosh
    class MockSchema:
        def __init__(self, **fields):
            self.fields = fields
    
    class MockText:
        def __init__(self, analyzer=None, stored=True):
            self.analyzer = analyzer
            self.stored = stored
    
    class MockID:
        def __init__(self, stored=True, unique=True):
            self.stored = stored
            self.unique = unique
    
    class MockStemmingAnalyzer:
        pass
    
    class MockIndex:
        def __init__(self, temp_dir, schema):
            self.temp_dir = temp_dir
            self.schema = schema
            self.docs = []
            
        def writer(self):
            return MockWriter(self)
            
        def searcher(self):
            return MockSearcher(self)
    
    class MockWriter:
        def __init__(self, index):
            self.index = index
            
        def add_document(self, **fields):
            self.index.docs.append(fields)
            
        def commit(self):
            pass
    
    class MockSearcher:
        def __init__(self, index):
            self.index = index
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def search(self, query, limit=10):
            results = []
            pattern = query.text
            for doc in self.index.docs:
                if pattern.lower() in doc['text'].lower():
                    results.append({'doc_id': doc['doc_id']})
                    if len(results) >= limit:
                        break
            return results
    
    class MockQueryParser:
        def __init__(self, field_name, schema):
            self.field_name = field_name
            self.schema = schema
            
        def parse(self, query_string):
            return MockQuery(query_string)
    
    class MockQuery:
        def __init__(self, text):
            self.text = text
    
    # Replace Whoosh classes with mocks
    Schema = MockSchema
    TEXT = MockText
    ID = MockID
    StemmingAnalyzer = MockStemmingAnalyzer
    
    class index:
        @staticmethod
        def create_in(temp_dir, schema):
            return MockIndex(temp_dir, schema)
    
    class qparser:
        QueryParser = MockQueryParser
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformer not found, using mock implementation")
    SentenceTransformer = MockSentenceTransformer

def load_subreddit_corpus(subreddit_name: str):
    """
    Loads a subreddit corpus from Convokit.

    Args:
        subreddit_name (str): The name of the subreddit corpus (e.g., 'subreddit-Cornell').

    Returns:
        Corpus: The loaded Convokit corpus object.
    """
    print(f"Downloading corpus for {subreddit_name}...")
    try:
        # Try using the real convokit download
        corpus = download(subreddit_name)
    except Exception as e:
        # If anything goes wrong, fall back to mock corpus
        print(f"Error downloading real corpus: {e}")
        print("Using mock corpus instead")
        corpus = MockCorpus(subreddit_name)
        
    print(f"Corpus loaded with {len(list(corpus.iter_utterances()))} utterances.")
    return corpus


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

def retrieve_lexical(lex_index, query: str, top_n=20):
    """
    Retrieves top-N documents from the lexical index for a given query.

    Args:
        lex_index (Any): The lexical index structure returned by build_lexical_index().
        query (str): The user's search query string.
        top_n (int): Number of top results to retrieve.

    Returns:
        List[Any]: A list of candidate documents (IDs or objects) retrieved from the index.
    """
    if 'whoosh.qparser' not in globals() and not WHOOSH_AVAILABLE:
        # Use our mock QueryParser
        parser = qparser.QueryParser("text", schema=lex_index.schema)
    else:
        # Import only if needed and not already imported
        if 'QueryParser' not in globals():
            from whoosh.qparser import QueryParser
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


def display_results(reranked_docs, corpus):
    """
    Displays the final search results to the user.

    Args:
        reranked_docs (List[Any]): The final list of documents after re-ranking.
        corpus (Corpus): The Convokit corpus containing the documents.

    Returns:
        None
    """
    if not reranked_docs:
        print("No results found.")
        return
    
    print(f"\nTop {len(reranked_docs)} results:")
    print("-" * 50)
    
    # Create a mapping of document IDs to utterances for quick lookup
    id_to_utterance = {str(utt.id): utt for utt in corpus.iter_utterances()}
    
    for i, doc_id in enumerate(reranked_docs, 1):
        if doc_id in id_to_utterance:
            utterance = id_to_utterance[doc_id]
            speaker = utterance.speaker.id if hasattr(utterance, 'speaker') else "Unknown"
            
            # Truncate text if it's too long
            text = utterance.text
            if len(text) > 200:
                text = text[:197] + "..."
                
            print(f"{i}. [{speaker}]: {text}")
            print("-" * 50)
        else:
            print(f"{i}. Document ID: {doc_id} (Not found in corpus)")
            print("-" * 50)

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
    parser = argparse.ArgumentParser(description="Hybrid Lexical + Semantic Search for a Subreddit Corpus.")
    parser.add_argument("--subreddit", type=str, help="Name of the subreddit corpus (e.g. 'subreddit-Cornell').")
    parser.add_argument("--query", type=str, help="Search query to run (exits after one search when provided).")
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

    # 5. Check if we have a query argument
    if args.query:
        query = args.query
        print(f"\nRunning search for query: {query}")
        
        # Retrieve lexical results
        candidate_docs = retrieve_lexical(lex_index, query)

        # Re-rank results
        reranked_docs = rerank_semantic(sem_index, candidate_docs, query)

        # Display results
        display_results(reranked_docs, corpus)
        return
    
    # 6. Query loop if no query argument was provided
    while True:
        try:
            query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting search...")
                break

            # Retrieve lexical results
            candidate_docs = retrieve_lexical(lex_index, query)

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

if __name__ == "__main__":
    main()
