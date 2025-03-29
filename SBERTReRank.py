import numpy as np
from collections import defaultdict
from itertools import islice
from sentence_transformers import SentenceTransformer, util
# SentenceTransformer library is a wrapper around BERT-based models
# Built for computing sentence embeddings instead of word-level embeddings like classic BERT
# https://huggingface.co/sentence-transformers

class SBERTReRanker:
    def __init__(self, ir_system, results_file, output_file):
        """
        Initializes the SBERT re-ranker.
        
        :param ir_system: An instance of our IRsystem.py class containing self.docs and self.queries
        :param results_file: The file containing initial ranking results (no re-ranking)
        :param output_file: The output file for the re-ranked results
        """
        self.ir_system = ir_system
        self.results_file = results_file
        self.output_file = output_file
        # Our SBERT Model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Our intitial results (dictionary of top n docs for each query)
        self.initial_results = defaultdict(set)

        with open(self.results_file, "r") as file:
            for line in file:
                parts = line.strip().split()
                query_id, _, doc_id, _, score, _ = parts
                self.initial_results[query_id].add((doc_id, float(score)))

    # Re-rank documents using SBERT and save to output file
    def re_rank(self):
        reranked_results = []

        # Compute query and doc embeddings using our SBERT model
        # Computing them all here helps avoid possibly redudant embedding calculations
        # This takes a while but saves time during similarity comparison
        print("\nGenerating Query Embeddings")
        query_ids = list(self.ir_system.queries.keys())
        query_texts = [f"{self.ir_system.queries[qid]}" for qid in query_ids]
        query_embeddings = {}

        query_vectors = self.model.encode(
            query_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        query_embeddings = dict(zip(query_ids, query_vectors))


        print("\nGenerating Document Embeddings")
        doc_ids = list(self.ir_system.docs.keys())
        doc_texts = [f"{self.ir_system.docs[doc_id]}" for doc_id in doc_ids]
        doc_embeddings = {}

        doc_vectors = self.model.encode(
            doc_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        doc_embeddings = dict(zip(doc_ids, doc_vectors))

        # For each query embedding, compute cosine similarity with each relevant doc embedding
        # For quick testing, can do: for query_id in dict(islice(query_embeddings.items(), 5)):
        print("\nChecking Query-Document Cosine Similarities")
        queries_processed = 0
        total_queries = len(self.ir_system.queries)
        for query_id in query_embeddings:
            query_embedding = query_embeddings[query_id]
            doc_scores = []

            for doc_id, _ in self.initial_results[str(query_id)]:
                doc_embedding = doc_embeddings[doc_id]
                similarity = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
                doc_scores.append((doc_id, similarity))

            # Sort documents by similarity x[1] = (id, sim)[1] where higher is better
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Store new rankings
            for rank, (doc_id, score) in enumerate(doc_scores, start=1):
                reranked_results.append(f"{query_id} Q0 {doc_id} {rank} {score:.4f} SBERT\n")
            
            queries_processed += 1
            if queries_processed % 100 == 0 or queries_processed == total_queries:
                print(f"{queries_processed} / {total_queries} queries")

        # Save to output file
        with open(self.output_file, "w") as output_file:
            output_file.writelines(reranked_results)

        print(f"SBERT re-ranking complete! Results saved to {self.output_file}")