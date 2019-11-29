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

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    words = get_lexicon()
    stop_words = set(list(STOP_WORDS) + ["know", "don", "ve", "say", "way", "said", "ll", "think", "thing", "don’t", "like", "got", "people", "going", "talk", "right", "happened", ">>"])
    print("Stop words", stop_words)
    
    doc_idxs = range(10) #### TODO: pick a reasonable number before 250

    # Create stemmer
    stemmer = WordNetLemmatizer() 
   
    results = []
    with CaptionIndex(idx_path, lexicon, documents) as index:
        for doc_id in tqdm.tqdm(doc_idxs):
            count = 1
            
            timeline = timeline_gender(str(doc_id))
            print("timeline done")
            if not timeline.shape[0]:
                continue
            postings = index.intervals(int(doc_id))
            
            sentence = ""
            starttime = None
            for p in postings:
                if starttime is None:
                    starttime = p.start

                # Cut after 30s
                if p.end - starttime > 30*count:
                    #import pdb; pdb.set_trace()
                    if not (timeline[int(starttime):min(int(p.end), len(timeline))] == 0).all():
                        results.append((sentence, 
                            np.sum(timeline[int(starttime):min(int(p.end), len(timeline)), 0]), 
                            np.sum(timeline[int(starttime):min(int(p.end), len(timeline)), 1]), 
                            np.mean(timeline[int(starttime):min(int(p.end), len(timeline)), 0]), 
                            np.mean(timeline[int(starttime):min(int(p.end), len(timeline)), 1]) ))
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
                    if word not in stop_words and len(word)>1:
                        stemmed_word = stemmer.lemmatize(word)
                        sentence += stemmed_word + " " 
    return results

if __name__ == '__main__':
    gen(**vars(get_args()))
