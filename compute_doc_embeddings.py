from wikipedia2vec import Wikipedia2Vec
import numpy as np
import os

wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

def get_doc_embedding(file_name):
    with open(file_name, 'r') as open_file:
        sum_vector = np.zeros(100)
        word_count = 0
        for line in open_file:
            split_file = line.split()
            
            
            for word in split_file:
                try:
                    word_embedding = wiki2vec.get_word_vector(word)
                    sum_vector += word_embedding
                    word_count += 1
                except KeyError:
                    continue
        open_file.close()
        return sum_vector / word_count


for filename in os.listdir('documents'):
    f = os.path.join('documents', filename)
    # if it's a file, compute doc embedding
    if f.endswith('.txt'):
        print(f)
        embedding = get_doc_embedding(f)
        np.savetxt(f + '.vec', embedding)
