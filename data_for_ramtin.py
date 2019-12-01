import numpy as np
import argparse
import os
import time
import traceback
from termcolor import colored, cprint
import pickle
import tqdm
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 

from utils import get_lexicon, get_channels 
from captions import Lexicon, Documents, CaptionIndex
from captions.query import Query
from captions.util import PostingUtil

<<<<<<< HEAD
import pickle

from test_face_gender import timeline_gender
DEFAULT_CONTEXT = 3
=======
from scan_face_gender import timeline_gender
>>>>>>> 9a7c722128d8519466f85c982d72c976b504aaaa

DEFAULT_CONTEXT = 30

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index_dir', type=str,
                        help='Directory containing index files')
    parser.add_argument('-c', dest='window_size', type=int,
                        default=DEFAULT_CONTEXT,
                        help='Context window width (default: {})'.format(
                             DEFAULT_CONTEXT))

    return parser.parse_args()



def data_generator(index_dir, window_size, include_stop_words=False):
    """Given a directory and a window size outputs a list of
    (sentence, number of men on screen, 
               number of women on screen,
               mean number of men on screen, 
               mean number of women on screen, 
               channel)

    sentence can be with or without stopwords
    """

    # Open the transcript files
    doc_path = os.path.join(index_dir, 'docs.list')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    channel = 'MSNBC'
    var = {'CNN':(1, 82529), 'FOX': (82530, 162639), 'MSNBC': (162640, 246922)}
    SIZE = 20000

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    # Getting words
    words = get_lexicon()
<<<<<<< HEAD
    stop_words = set([">>"])
    print("Stop words", stop_words)
=======

    # Getting channels
    docid_to_channels = get_channels()

    # Selecting stop words
    stop_words = set(list(STOP_WORDS) + ["know", "don", "ve", "say", "way", "said", "ll", "think", "thing", "don’t", "like", "got", "people", "going", "talk", "right", "happened", ">>"])
>>>>>>> 9a7c722128d8519466f85c982d72c976b504aaaa
    
    start_idx, end_idx = var[channel] 
    doc_idxs = list(np.random.choice(np.arange(start_idx, end_idx), SIZE))

    # Create stemmer
<<<<<<< HEAD
    # stemmer = WordNetLemmatizer() 
   
    with CaptionIndex(idx_path, lexicon, documents) as index:
        for doc_id in tqdm.tqdm(doc_idxs):
            results = {}

            count = 1
            
            gender, locations, persons = timeline_gender(str(doc_id))
            #print("meta data extracted")

            if len(gender.keys()) == 0:
                #print("Skipped id %d"%(doc_id))
                continue

=======
    stemmer = WordNetLemmatizer() 

    # Container for result tuples
    results = []
    with CaptionIndex(idx_path, lexicon, documents) as index:
        for doc_id in tqdm.tqdm(doc_idxs):
            ## Get channel
            channel = docid_to_channels[doc_id]

            count = 1
            
            # Loading the timeline of faces and their gender
            timeline = timeline_gender(str(doc_id))
            # If no faces in doc
            if not timeline.shape[0]:
                continue

            # Get all the transcripts 
>>>>>>> 9a7c722128d8519466f85c982d72c976b504aaaa
            postings = index.intervals(int(doc_id))
            
            sentence = ""
            starttime = None
            for p in postings:
                if starttime is None:
                    starttime = p.start

<<<<<<< HEAD
                # Cut after 30s
                if p.end - starttime > 3*count:
                    #import pdb; pdb.set_trace()
                    t1 = int(starttime)
                    t2 = int(p.end)
                    for time_box in range(t1, t2):
                        if time_box in gender.keys():
                            frame_gender = gender[time_box]
                        else:
                            frame_gender = None

                        if time_box in persons.keys():
                            frame_persons = persons[time_box]
                        else:
                            frame_persons = None

                        if time_box in locations.keys():
                            frame_loc = locations[time_box]
                        else:
                            frame_loc = None

                        if time_box in results.keys():
                            results[time_box].append({'text': sentence, 'gender': frame_gender,
                                                     'persons': frame_persons, 'locations': frame_loc})
                        else:
                            results[time_box] = [{'text': sentence, 'gender': frame_gender,
                                                  'persons': frame_persons, 'locations': frame_loc}]

                    # if not (timeline[int(starttime):min(int(p.end), len(timeline))] == 0).all():
                    #     results.append((sentence, 
                    #         np.sum(timeline[int(starttime):min(int(p.end), len(timeline)), 0]), 
                    #         np.sum(timeline[int(starttime):min(int(p.end), len(timeline)), 1]), 
                    #         np.mean(timeline[int(starttime):min(int(p.end), len(timeline)), 0]), 
                    #         np.mean(timeline[int(starttime):min(int(p.end), len(timeline)), 1]) ))
=======
                # Cut after windows_size s
                if p.end - starttime > window_size*count:
                    t1 = int(starttime)
                    t2 = min(int(p.end), len(timeline))
                    # Check if any faces appear in the timeframe
                    if not (timeline[t1:t2] == 0).all():
                        male_timeline = timeline[t1:t2, 0]
                        female_timeline = timeline[t1:t2, 1]
                        results.append((sentence, 
                            np.sum(male_timeline),
                            np.sum(female_timeline), 
                            np.mean(male_timeline), 
                            np.mean(female_timeline), channel))
                    # Start a new sentence
>>>>>>> 9a7c722128d8519466f85c982d72c976b504aaaa
                    count += 1
                    starttime = p.end
                    sentence = ""

                # Get words in posting
                tokens = index.tokens(0, p.idx, p.len)
                if not tokens:
                    continue
                for token in tokens:
                    # Getting corresponding word
                    word = words[token]
<<<<<<< HEAD
                    # stemmed_word = stemmer.stem(word)
                    #if word not in stop_words and len(word)>1:
                        #stemmed_word = stemmer.lemmatize(word)
                    if word not in stop_words:
                        sentence += word + " " 
            pickle.dump(results, open('%s/meta_data_%d.p'%(channel, doc_id), 'wb'))
    return None
=======
                    # Add word if we want all stopwords or if not a stopword 
                    if include_stop_words or (word not in stop_words and len(word)>1):
                        stemmed_word = stemmer.lemmatize(word)
                        sentence += stemmed_word + " " 
    return results
>>>>>>> 9a7c722128d8519466f85c982d72c976b504aaaa


## DEPRECATED
def gen(index_dir, silent, windows_size):
    return [x[0:5] for x in data_generator(index_dir, windows_size, False)]

if __name__ == '__main__':
    args = get_args()
    for x in data_generator(args.index_dir, args.window_size, True):
        print(x)
