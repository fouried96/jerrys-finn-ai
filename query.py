from src.rag import query_faiss

# Define your query
question = "How do I use the Waves CLA-76 compressor for pop punk vocals?"

# Retrieve and print the response
response = query_faiss(question)
print(response)