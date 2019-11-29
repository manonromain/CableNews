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

from test_face_gender import gender_to_time
DEFAULT_CONTEXT = 3

gender_reqs = {"msup":0, "fsup":20, "finf":1, "minf":0}

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
    parser.add_argument('-f', dest='folder', help='where to save')
    parser.add_argument('--use_gender', action='store_true')

    return parser.parse_args()



def main(index_dir, silent, context_size, folder, use_gender):
    doc_path = os.path.join(index_dir, 'docs.list')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    words = get_lexicon()
    stop_words = set(list(STOP_WORDS) + ["know", "don", "ve", "say", "way", "said", "ll", "think", "thing", "don’t", "like", "got", "people", "going", "talk", "right", "happened", ">>"])
    print("Stop words", stop_words)
    
    doc_idxs = range(144, 246923)
    word_idx_dic = {}
    idx_counter = 0

    # Create folder
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Create stemmer
    stemmer = WordNetLemmatizer() 
    with CaptionIndex(idx_path, lexicon, documents) as index:
        for doc_id in tqdm.tqdm(doc_idxs):
            dic = {}
            count = 1
            if use_gender:
                intervals_gender = gender_to_time(str(doc_id), gender_reqs)
                postings = []
                for t1, t2 in intervals_gender:
                    postings.extend(index.intervals(int(doc_id), t1, t2))
            else:
                postings = index.intervals(int(doc_id))
            
            starttime = None

            for p in postings:
                if starttime is None:
                    starttime = p.start

                # Cut after 30s
                if p.end - starttime > 30*count:
                    pickle.dump(dic, open(os.path.join(folder, 'Doc_%d_Chunk_%d.p'%(doc_id, count-1)),'wb'))
                    dic = {}
                    count += 1
                    starttime = p.end

                # Get words in posting
                tokens = index.tokens(0, p.idx, p.len)
                if not tokens:
                    continue
                for token in tokens:
                    word = words[token]
                    # stemmed_word = stemmer.stem(word)
                    if word not in stop_words and len(word)>1:
                        stemmed_word = stemmer.lemmatize(word)
                        # print("Word {} -> {}".format(word, stemmed_word))
                        if stemmed_word not in word_idx_dic.keys():
                            word_idx_dic[stemmed_word] = idx_counter
                            idx_counter += 1
                        idx_token = word_idx_dic[stemmed_word]
                        if idx_token in dic:
                            dic[idx_token] += 1
                        else:
                            dic[idx_token] = 1
    pickle.dump(word_idx_dic, open(os.path.join(folder, "word_idx.p"), "wb"))

if __name__ == '__main__':
    main(**vars(get_args()))
