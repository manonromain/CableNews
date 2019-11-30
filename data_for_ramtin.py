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

from utils import get_lexicon
from captions import Lexicon, Documents, CaptionIndex
from captions.query import Query
from captions.util import PostingUtil

import pickle

from test_face_gender import timeline_gender
DEFAULT_CONTEXT = 3

#gender_reqs = {"msup":0, "fsup":20, "finf":1, "minf":0}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index_dir', type=str,
                        help='Directory containing index files')
    parser.add_argument('-s', dest='silent', action='store_true',
                        help='Silent mode for benchmarking')
    parser.add_argument('-c', dest='context_size', type=int,
                        default=DEFAULT_CONTEXT,
                        help='Context window width (default: {})'.format(
                             DEFAULT_CONTEXT))

    return parser.parse_args()



def gen(index_dir, silent, context_size):
    doc_path = os.path.join(index_dir, 'docs.list')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    channel = 'MSNBC'
    var = {'CNN':(1, 82529), 'FOX': (82530, 162639), 'MSNBC': (162640, 246922)}
    SIZE = 20000

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    words = get_lexicon()
    stop_words = set([">>"])
    print("Stop words", stop_words)
    
    start_idx, end_idx = var[channel] 
    doc_idxs = list(np.random.choice(np.arange(start_idx, end_idx), SIZE))

    # Create stemmer
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

            postings = index.intervals(int(doc_id))
            
            sentence = ""
            starttime = None
            for p in postings:
                if starttime is None:
                    starttime = p.start

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
                    count += 1
                    starttime = p.end
                    sentence = ""

                # Get words in posting
                tokens = index.tokens(0, p.idx, p.len)
                if not tokens:
                    continue
                for token in tokens:
                    word = words[token]
                    # stemmed_word = stemmer.stem(word)
                    #if word not in stop_words and len(word)>1:
                        #stemmed_word = stemmer.lemmatize(word)
                    if word not in stop_words:
                        sentence += word + " " 
            pickle.dump(results, open('%s/meta_data_%d.p'%(channel, doc_id), 'wb'))
    return None

if __name__ == '__main__':
    gen(**vars(get_args()))
