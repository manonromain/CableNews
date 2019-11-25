import numpy as np
import pickle
from scipy.sparse import coo_matrix
from sklearn.decomposition import LatentDirichletAllocation
import glob
import tqdm
import lda
import os
from utils import get_lexicon
from nltk.stem import WordNetLemmatizer 
from termcolor import colored
from seed_topics import seed_topics
import guidedlda

word2idx = pickle.load(open("ECJ_gendered/word_idx.p", "rb"))
print(word2idx)
# Load seed topics
seed_topics, topics = seed_topics(word2idx)

idx_to_word = {v: k for k, v in word2idx.items()}
# Load data
print("Starting training...")
lda = guidedlda.GuidedLDA(n_topics=len(topics), n_iter=100, random_state=7, refresh=20)


## Concat data
row, col, data = np.array(()), np.array(()), np.array(())

matrix_data_list = glob.glob("ECJ_gendered/matrix_data_*.p")
np.random.shuffle(matrix_data_list)
for doc in tqdm.tqdm(matrix_data_list):
    print("Partial fitting", doc)
    res = pickle.load(open(doc, "rb"))
    row = np.append(row, np.int32(res["I"]))
    col = np.append(col, np.int32(res["J"]))
    data = np.append(data, np.int32(res["data"]))
    X = coo_matrix((np.int32(data), (np.int32(row), np.int32(col))))
    
lda.fit(X, seed_topics=seed_topics, seed_confidence=0.4)

print("Training done")
def print_top_words(model, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #{} - {}: ".format(topic_idx, topics[topic_idx])
        message += " ".join([idx_to_word[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

def print_sentence_and_topic(sentence, topic):
    print(colored("Sentence:", "blue"), colored(sentence, "green"))
    print(colored("Topic:   ", "blue"), colored(topic, "red"))

print_top_words(lda, 20)
np.save(open("ECJ_gendered/{}_components.npy".format(MODEL), "wb"), lda.components_)

## Test for input sentences
stemmer = WordNetLemmatizer() 
while True:
    sentence = input()
    list_words = [w.lower() for w in sentence.split()]
    np_array = np.zeros([1, len(word2idx.keys())])
    for word in list_words:
        stemmed_word = stemmer.lemmatize(word)
        if stemmed_word in word2idx:
            print(stemmed_word)
            np_array[0, word2idx[stemmed_word]] += 1
    topic_dist = lda.transform(np.int32(np_array))
    print_sentence_and_topic(sentence, topics[np.argmax(topic_dist)])
