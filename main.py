from IRsystem import IRsystem # import IRsystem class


# Code to adjust scifact test.tsv to include q0 column
# input_file = './scifact/qrels/test.tsv'
# output_file = './scifact/qrels/modified_test.tsv'
# with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#     for line in infile:
#         parts = line.strip().split('\t')
        
#         if len(parts) == 3:
#             query_id, corpus_id, score = parts
#             outfile.write(f"{query_id}\tq0\t{corpus_id}\t{score}\n") 


# Example text processing
# query = "<p>This is an example query, with numbers 123 and punctuation!?!?!</p>"
# print(IR.preprocess(query))

# Example index build (print most common words and how many docs they appear in)
# IR.build_index("./scifact/corpus.jsonl")
# index = IR.get_index()
# for key, value in index.items() :
#     if len(value) > 1500:
#         print(key, len(value))


# Run system to generate output
#   - Use only test queries (the queries with odd numbers 1.3.5, …)
#   - Each query has 100 documents in the decending rank order
#   - Output Results.txt (and ResultsTitlesOnly.txt for index that excludes document text)
IR = IRsystem()
IR.build_index("./scifact/corpus.jsonl", False)
IR.save_results("scifact/queries.jsonl", "Results.txt")

# Display sample results
# IR.display_samples("scifact/queries.jsonl") # default display top 10