import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

class E5ReRanker:
    def __init__(self, ir_system, results_file, output_file):
        """
        Initializes the E5 re-ranker using intfloat/e5-large-v2.

        :param ir_system: An instance of IRsystem.py with self.docs and self.queries
        :param results_file: Initial ranking file (top 100 docs per query)
        :param output_file: Output file to store re-ranked results
        """
        self.ir_system = ir_system
        self.results_file = results_file
        self.output_file = output_file

        # Load the E5 model
        print("Loading intfloat/e5-large-v2 model...")
        print("Using GPU with cuda" if torch.cuda.is_available() else "Using CPU since cuda not found")
        self.model = SentenceTransformer("intfloat/e5-large-v2", device='cuda' if torch.cuda.is_available() else 'cpu')

        # Load initial IR results (top 100 per query)
        self.initial_results = defaultdict(set)
        with open(self.results_file, "r") as file:
            for line in file:
                parts = line.strip().split()
                query_id, _, doc_id, _, score, _ = parts
                self.initial_results[query_id].add((doc_id, float(score)))
        
        self.query_embeddings = {}
        self.doc_embeddings = {}
        self.total_queries = len(self.ir_system.queries)

    def start_embed(self):
        """
        Batch embeds all queries and documents from the IRsystem using E5 model.
        """
        # ============================= Query Embedding ================================
        # Query embeddings start (Extract ids, texts with added prefix)
        print("\n[Non-dynamic batch] Query embeddings start...")
        query_ids = list(self.ir_system.queries.keys())
        query_texts = [f"query: {self.ir_system.queries[qid]}" for qid in query_ids]  # Apply E5 prefix
        
        # Batch encode with E5 (faster + efficient)
        print("\nQuery batch encoding with E5...")
        query_vectors = self.model.encode(
            query_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        # Store in dictionary for later re-ranking
        print("\nEncoding complete, query dictionary storing...")
        self.query_embeddings = dict(zip(query_ids, query_vectors))

        # Query embbeding complete 
        print(f" {len(self.query_embeddings)} query embeddings generated.")

        # ============================= Doc Embedding ================================
        # Doc embeddings start (Extract ids, texts with added prefix)
        print("\n[Non-dynamic batch] Doc embeddings start...")
        doc_ids = list(self.ir_system.docs.keys())
        doc_texts = [f"passage: {self.ir_system.docs[doc_id]}" for doc_id in doc_ids]

        # Batch encode all documents with E5
        print("\nDoc batch encoding with E5...")
        doc_vectors = self.model.encode(
            doc_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        print("\nEncoding complete, doc dictionary storing...")
        # Store results in dictionary
        self.doc_embeddings = dict(zip(doc_ids, doc_vectors))

        # Doc embbeding complete
        print(f"{len(self.doc_embeddings)} document embeddings generated.")

    def re_rank(self):
        reranked_results = []
        queries_processed = 0

        # Compute cosine similarity and re-rank
        # For each query embedding, compute cosine similarity with each relevant doc embedding
        print("\nComputing similarities and re-ranking...")
        # for query_id in self.query_embeddings:
        for query_id in self.query_embeddings:
            query_vec = self.query_embeddings[query_id]
            doc_scores = [] 

            for doc_id, _ in self.initial_results[str(query_id)]:
                doc_vec = self.doc_embeddings[doc_id]
                similarity = util.pytorch_cos_sim(query_vec, doc_vec).item()
                doc_scores.append((doc_id, similarity))

            # Sort documents by similarity score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Store the new ranking
            for rank, (doc_id, score) in enumerate(doc_scores, start=1):
                reranked_results.append(f"{query_id} Q0 {doc_id} {rank} {score:.4f} E5\n")

            queries_processed += 1
            if queries_processed % 100 == 0 or queries_processed == self.total_queries:
                print(f"{queries_processed} / {self.total_queries} queries")

        # Save re-ranked results to output file
        with open(self.output_file, "w") as output_file:
            output_file.writelines(reranked_results)

        print(f"\n E5 re-ranking complete! Results saved to {self.output_file}")
