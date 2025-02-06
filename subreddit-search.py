#!/usr/bin/env python3
"""
Hybrid Lexical + Semantic Search for a Subreddit Corpus.

Usage:
    python search_reddit.py --subreddit subreddit-NameHere
"""
import argparse

from convokit import Corpus, download

def load_subreddit_corpus(subreddit_name: str):
    pass


def preprocess_corpus(corpus):
    pass


def build_lexical_index(corpus):
    pass


def build_semantic_index(corpus):
    pass


def retrieve_lexical(lex_index, query: str):
    pass


def rerank_semantic(sem_index, candidate_docs, query: str):
    pass


def display_results(reranked_docs):
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
    pass

if __name__ == "__main__":
    main()
