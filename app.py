from wikipedia2vec import Wikipedia2Vec
wiki2vec = Wikipedia2Vec.load('pretrained_model.txt')
print(wiki2vec.get_word_vector('the'))
# Compile list of documents to search
# Compute their doc embeddings

# Implement a way to gather query embedding from client
# Search through doc embeddings linearly