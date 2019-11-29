import glob
import os 
from nltk.stem import WordNetLemmatizer 


def seed_topics(word2idx):
    seed_topics = {}
    topic_id = 0
    stemmer = WordNetLemmatizer() 
    topics = []
    next_topic = True
    found_single_word_in_topic = True
    with open("topics.txt", "rt") as file_txt:
        for line in file_txt.readlines():
            word = line.rstrip()
            if not word:
                if not found_single_word_in_topic:
                    topics[-1] = ""
                topic_id += 1
                next_topic = True
                continue
            if next_topic:
                next_topic = False
                found_single_word_in_topic = False
                topics.append(word)
                continue
            word = stemmer.lemmatize(word.lower())
            if word not in word2idx:
                print(word, "not in vocabulary")
                continue
            found_single_word_in_topic = True
            seed_topics[word2idx[word]] = topic_id
    print("We have", topic_id + 1, "topics")
    return seed_topics, topics
