import numpy as np
import pickle
from scipy.sparse import coo_matrix
from sklearn.decomposition import LatentDirichletAllocation
import glob
import tqdm
import lda
from utils import get_lexicon
from nltk.stem import WordNetLemmatizer 
from termcolor import colored


word2idx = pickle.load(open("ECJ_doc_30s/word_idx.p", "rb"))
idx_to_word = {v: k for k, v in word2idx.items()}
def print_top_words(components, n_top_words):
    for topic_idx, topic in enumerate(components):
        message = "Topic #%d: " % topic_idx
        message += " ".join([idx_to_word[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

def print_sentence_and_topic(sentence, topic):
    print(colored("Sentence:", "blue"), colored(sentence, "green"))
    print(colored("Topic:   ", "blue"), colored(topic, "red"))


for model in ["lda"]:
    print(model)
    components = np.load("ECJ_doc_30s/{}_components.npy".format(model), allow_pickle=True)
    print_top_words(components, 10)
    topics = []
    for topic_idx, topic in enumerate(components):
        name = input("Name of topic {}:".format(topic_idx)) 
        topics.append(name)
    
    stemmer = WordNetLemmatizer() 
    while True:
        sentence = input()
        list_words = [w.lower() for w in sentence.split()]
        np_array = np.zeros([1, len(word2idx.keys())])
        for word in list_words:
            stemmed_word = stemmer.lemmatize(word)
            if stemmed_word in word2idx:
                np_array[0, word2idx[stemmed_word]] += 1
        topic_dist = lda.transform(np.int32(np_array))
        print_sentence_and_topic(sentence, topics[np.argmax(topic_dist)])
    
