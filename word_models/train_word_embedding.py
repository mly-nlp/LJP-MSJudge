from keras.preprocessing.text import Tokenizer
import joblib
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec


file_path = "../data/judgment_prediction_dataset.json"

with open(file_path, 'r', encoding="utf-8") as load_f:
    data = json.load(load_f)

keys = list(data.keys())
court_debates =[] 

for key in keys:
    dialogue = data[key]["court_debate"]
    court_debates.append(dialogue)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(court_debates)
joblib.dump(tokenizer, './tokenizer.pkl') # 模型保存

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
sentences = [s.split(" ") for s in court_debates]
EMBEDDING_DIM = 300
word2vec_model = Word2Vec(min_count=0, size=EMBEDDING_DIM, sg=1, hs=0, negative=6, iter=10, workers=64, window=5, seed=6)
word2vec_model.build_vocab(sentences)
print("num of sentences:", len(sentences))
word2vec_model.train(sentences, total_examples=len(sentences), epochs=word2vec_model.iter)
word2vec_model.save("./word2vec.model")# 模型保存
# word2vec_model = Word2Vec.load("./new_models/word2vec.model")

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = word2vec_model[word]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)
np.save('./embedding_matrix.npy',embedding_matrix) 