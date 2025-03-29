import pickle
from SBERTReRank import SBERTReRanker

# Load the saved IR (can be modified in store_IR.py)
with open("ir_system.pkl", "rb") as f:
    IR = pickle.load(f)

# Re-rank using SBERT
#   - Input our IR system (for queries and document processing) and Base results (for re-ranking)
#   - Output SBERT_Results.txt
print("\nBuilding SBERT model to re-rank results")
reranker = SBERTReRanker(IR, "Base_Results.txt", "SBERT_Results.txt")
reranker.re_rank()


# Command for generating MAP scores using trec_eval based on a results file i.e. for Base_Results.txt:
# ./trec_eval-9.0.7/trec_eval -q -m map ./scifact/qrels/modified_test.tsv ./Base_Results.txt > Base_MapScores.txt

# And command for Precision at 10 scores using trec_eval (Base example)
# ./trec_eval-9.0.7/trec_eval -q -m P.10 ./scifact/qrels/modified_test.tsv ./Base_Results.txt > Base_PrecisionAt10.txt

