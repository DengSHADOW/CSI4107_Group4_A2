from E5ReRank import E5ReRanker
import pickle

# Load the saved IR (can be modified in store_IR.py)
with open("ir_system.pkl", "rb") as f:
    IR = pickle.load(f)

# Re-rank using E5
#   - Input our IR system (for queries and document processing) and Base results (for re-ranking)
#   - Output E5_Results.txt
print("\nIR loading complete, building E5 model to re-rank results")
E5ranker = E5ReRanker(IR, "Base_Results.txt", "E5_Results.txt")

# Embedding and re-ranking 
E5ranker.start_embed()
E5ranker.re_rank()

# Command for generating MAP scores using trec_eval based on a results file i.e. for Base_Results.txt:
# ./trec_eval-9.0.7/trec_eval -q -m map ./scifact/qrels/modified_test.tsv ./Base_Results.txt > Base_MapScores.txt

# And command for Precision at 10 scores using trec_eval (Base example)
# ./trec_eval-9.0.7/trec_eval -q -m P.10 ./scifact/qrels/modified_test.tsv ./Base_Results.txt > Base_PrecisionAt10.txt

