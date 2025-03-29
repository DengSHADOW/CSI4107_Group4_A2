import pickle
from IRsystem import IRsystem

# Build IR system as .pkl file (full object storage)
# Run system to generate output
#   - Use only test queries (the queries with odd numbers 1.3.5, …)
#   - Each query has 100 documents in the decending rank order
#   - Output Base_Results.txt
IR = IRsystem()
IR.build_index("./scifact/corpus.jsonl")
IR.save_results("scifact/queries.jsonl", "Base_Results.txt")

# Store it as .pkl file for later use
print("IR building complete, now start storing")
with open("IR_system.pkl", "wb") as f:
    pickle.dump(IR, f)

print("IRsystem saved to ir_system.pkl")
