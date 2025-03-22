import re
import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class IRsystem:
    """
    Information Retrieval System using TF-IDF and Cosine Similarity.

    This class:
    - Loads and processes text (removing stopwords, punctuation).
    - Builds an inverted index mapping words to document IDs.
    - Stores cleaned document texts for efficient retrieval.
    - Performs document ranking using TF-IDF and Cosine Similarity.
    - Saves search results in TREC format.

    Attributes:
        stop_words (set): Set of stopwords for text cleaning.
        index (dict): Inverted index mapping words to document IDs.
        doc_texts (dict): Stores cleaned document texts.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer instance.
    """

    def __init__(self, stopwords_file="StopWords.txt"):
        """
        Initialize the class, loading stopwords if provided.

        Args:
            stopwords_file (str): Path to stopwords file (optional).
        """
        # Load stopwords from StopWords.txt
        self.stop_words = set()
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as file:
                self.stop_words = set(word.strip().lower() for word in file.readlines())
        except FileNotFoundError:
            print("Stopwords file not found. Using an empty stopwords list.")
        
        # Empty Inverted index
        self.index = defaultdict(set)

        # Empty cleaned text dict (for retrieval) {Id: texts}
        self.docs = {}  
        # Same for queries
        self.queries = {}

        # Initialize vectorizer for calculating TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words=None)

    def preprocess(self, text):
        """
        Cleans and tokenizes text by:
        - Removing HTML tags
        - Removing punctuation and numbers
        - Lowercasing
        - Removing stopwords

        Args:
            text (str): Raw text input.

        Returns:
            list: List of cleaned tokens.
        """
        text = re.sub(r'(<[^>]+>|[^a-zA-Z\s])', ' ', text)  # Remove HTML tags, punctuation, and numbers
        tokens = text.lower().split()  # Tokenize

        # Remove stopwords and duplicate tokens
        return [word for word in tokens if word not in self.stop_words]

    def build_index(self, corpus_file):
        """
        Builds an inverted index and stores document texts for retrieval.

        Args:
            corpus_file (str): Path to corpus.jsonl file.

        Returns:
            None (Modifies self.index and self.docs)
        """
        with open(corpus_file, 'r', encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line)
                doc_id = doc["_id"]
                text = doc.get("title", "") + " " + doc.get("text", "")
                
                # Preprocess text
                words = self.preprocess(text)
                self.docs[doc_id] = " ".join(words)  # Store cleaned text

                # Build inverted index
                for word in words:
                    self.index[word].add(doc_id)

        print("Building Inverted indexing complete. Totally ", len(self.docs), "documents.")

    def RetrievalAndRanking(self, query_text):
        """
        Searches for relevant documents using TF-IDF and cosine similarity (using sklearn).

        Args:
            query_text (str): Raw user query.
            top_k (int): Number of top-ranked results.

        Returns:
            list: Ranked (doc_id, score) pairs.
        """

        # Preprocess the raw query
        query_tokens = self.preprocess(query_text)  

        # Find and retrieve relevant docs using index
        relevant_docs = set()
        for token in query_tokens:
            if token in self.index:
                relevant_docs.update(self.index[token])
        
        if not relevant_docs: # If no docs relevant
            return []

        unique_docs = list(relevant_docs)  # Convert set to list for indexing
        doc_text_list = [self.docs[doc_id] for doc_id in unique_docs if doc_id in self.docs]

        # Apply TF-IDF Vectorization only on relevant documents 
        tfidf_matrix = self.vectorizer.fit_transform(doc_text_list) # Matrix for docs 
        query_vec = self.vectorizer.transform([" ".join(query_tokens)]) # Vector for processed query

        # Compute cosine similarity
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Rank documents
        ranked_indices = np.argsort(similarity_scores)[::-1]  # Sort descending
        ranked_results = [(unique_docs[i], similarity_scores[i]) for i in ranked_indices]  

        return ranked_results
    
    def save_results(self, query_file, output_file, relevance_file="test.tsv", top_k=100):
        """
        Searches queries (only odd-numbered) and saves results in required format.

        Args:
            query_file (str): JSON file containing test queries.
            output_file (str): Output file to save results.
            relevance_file (str): Path to the TREC relevance judgments file.
            top_k (int): Max results per query.
        """
        # Ask user for a tag
        tag = input("Enter a tag for this run: ").strip()
        if not tag:
            tag = "default_run"  # Use a default tag if input is empty

        with open(query_file, 'r', encoding='utf-8') as f, open(output_file, 'w') as out:
            print("Searching and running, please wait......")
            for line in f:
                query = json.loads(line)
                query_id = int(query['_id'])  # Convert to integer
                if query_id % 2 == 0:  # Skip even-numbered queries
                    continue
                query_text = query['text']  # Extract query content
                self.queries[query_id] = query_text

                ranked_docs = self.RetrievalAndRanking(query_text)  

                for rank, (doc_id, score) in enumerate(ranked_docs[:top_k], start=1):
                    out.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {tag}\n")
                    
        print(f"Results saved to {output_file}")

    def display_samples(self, query_file, top_k=10):
        """
        Displays:
        1. Number of Vocabularies.
        1. First 100 tokens from the vocabulary.
        2. First 10 ranked results for the first 2 queries.

        Args:
            query_file (str): JSON file containing test queries.
            top_k (int): Number of top-ranked results to display per query.
        """

        # Print number of vocabularies
        print("\nTotal number of Vocabularies: ", len(self.index))
        # Print 100 sample tokens from vocabulary
        sample_tokens = list(self.index.keys())[:100]  # Get first 100 tokens
        print("\nSample 100 Tokens from Vocabulary:")
        print(", ".join(sample_tokens))

        first_two_queries = 2  # Track how many queries we've printed

        # Retrieve and print first 10 results for first 2 queries
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line)
                
                query_id = int(query['_id'])  # Convert to integer
                if query_id % 2 == 0:  # Skip even-numbered queries
                    continue
                
                query_text = query['text']
                ranked_docs = self.RetrievalAndRanking(query_text)  # Get all results

                if first_two_queries > 0:
                    print(f"\nFirst {top_k} Results for Query {query_id}: \n    \"{query_text}\"")
                    for rank, (doc_id, score) in enumerate(ranked_docs[:top_k], start=1):
                        print(f"{rank}. {doc_id} (Score: {score:.4f})")
                    first_two_queries -= 1

                if first_two_queries == 0:  # Stop after first 2 queries
                    break

    def get_index(self):
        """Returns the inverted index."""
        return self.index

    def get_docs(self):
        """Returns the stored document texts."""
        return self.docs
