# Hybrid Lexical + Semantic Search

This project is a **test** of combining a classic lexical search method (Whoosh/BM25) with a semantic re-ranker (Sentence-BERT) to improve the relevance of search results over Reddit data retrieved via [Convokit](https://convokit.cornell.edu/). Really, this is just a proof of concept to explore hybrid retrieval methods.

## Overview
- **Data Source**: A subreddit corpus loaded through Convokit's built-in download functionality.
- **Lexical Index**: We use [Whoosh](https://whoosh.readthedocs.io/en/latest/) to build a BM25-based index of each Reddit post/comment.
- **Semantic Index**: We use the [sentence-transformers](https://www.sbert.net/) library to create embeddings for each document.  
- **Hybrid Search Flow**:
  1. Retrieve top documents using lexical search.
  2. Re-rank those documents by semantic similarity using Sentence-BERT.

## How it works
1. **Load Data**: The program downloads a specific subreddit corpus using a name supplied by the user.
2. **Preprocess**: Text is lowercased and punctuations are removed.
3. **Build Indexes**: 
   - A Whoosh index (BM25) for lexical retrieval.
   - A Sentence-BERT embedding store for semantic re-ranking.
4. **Query**: The user types a query, which is first handled by the lexical index to get top results.
5. **Re-rank**: Those results are re-scored based on their similarity to the query’s embedding, producing final ranked results.
