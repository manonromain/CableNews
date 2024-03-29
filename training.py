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

MODEL = "lda" #"sklearn_batch"

word2idx = pickle.load(open("ECJ_gendered/word_idx.p", "rb"))
idx_to_word = {v: k for k, v in word2idx.items()}
# Load data
print("Starting training...")
if MODEL  == "sklearn":
    lda = LatentDirichletAllocation(total_samples=246913, n_jobs=4, verbose=1, n_components=15, random_state=1, learning_method="online", max_iter=100, learning_offset=50)
elif MODEL == "lda":
    lda = lda.LDA(n_topics = 20, n_iter=200, random_state=1)
else:
    lda = LatentDirichletAllocation(n_jobs=4, verbose=1, n_components=15, random_state=1, learning_method="batch", max_iter=100)
row, col, data = np.array(()), np.array(()), np.array(())

matrix_data_list = glob.glob("ECJ_gendered/matrix_data_*.p")
np.random.shuffle(matrix_data_list)
for doc in tqdm.tqdm(matrix_data_list):
    if MODEL == "sklearn":
        row, col, data = np.array(()), np.array(()), np.array(())
    print("Partial fitting", doc)
    res = pickle.load(open(doc, "rb"))
    row = np.append(row, np.int32(res["I"]))
    col = np.append(col, np.int32(res["J"]))
    data = np.append(data, np.int32(res["data"]))
    X = coo_matrix((np.int32(data), (np.int32(row), np.int32(col))))
    if MODEL == "sklearn":
        lda.partial_fit(X)
if MODEL != "sklearn":
    lda.fit(X)
#    break

print("Training done")
def print_top_words(model, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([idx_to_word[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

def print_sentence_and_topic(sentence, topic):
    print(colored("Sentence:", "blue"), colored(sentence, "green"))
    print(colored("Topic:   ", "blue"), colored(topic, "red"))

print_top_words(lda, 20)
np.save(open("ECJ_doc_30s/{}_components.npy".format(MODEL), "wb"), lda.components_)

topic_file_name = "ECJ_doc_30s/{}_topics.npy".format(MODEL)

if not os.path.exists(topic_file_name):
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        name = input("Name of topic {}:".format(topic_idx)) 
        topics.append(name)
    np.save(open(topic_file_name, "wb"), np.array(topics))
else:
    topics = np.load(topic_file_name)

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
