from wikipedia2vec import Wikipedia2Vec
import os
import numpy as np

wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')


def linearly_search_docs(query):
    query_embedding = wiki2vec.get_word_vector(query)
    max_similarity = None
    most_similar_file = ''

    for filename in os.listdir('documents'):
        if filename.endswith('.vec'):
            doc_embedding = np.loadtxt(os.path.join('documents', filename))
            similarity = cosine_similarity(query_embedding, doc_embedding)
            if max_similarity is None or similarity > max_similarity:
                max_similarity = similarity
                most_similar_file = filename
    return most_similar_file, max_similarity
            
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    
# Implement a way to gather query embedding from client
while True:
    client_input = input("Enter a query: ")
    print("Result: ", linearly_search_docs(client_input))


# Search through doc embeddings linearly
