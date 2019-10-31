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

from utils import get_lexicon
from captions import Lexicon, Documents, CaptionIndex
from captions.query import Query
from captions.util import PostingUtil
DEFAULT_CONTEXT = 3

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
    parser.add_argument('-i', dest='doc_id', type=int, help='doc_ic')

    return parser.parse_args()



def main(index_dir, silent, context_size, doc_id):
    doc_path = os.path.join(index_dir, 'docs.list')
    lex_path = os.path.join(index_dir, 'words.lex')
    idx_path = os.path.join(index_dir, 'index.bin')

    documents = Documents.load(doc_path)
    lexicon = Lexicon.load(lex_path)

    words = get_lexicon()
    stop_words_custom = set(list(STOP_WORDS) + [".", ",", ">>", ":", ";"])
    stop_words = [words.index(w) for w in stop_words_custom if w in words]
    print("Stop words", STOP_WORDS)
    bs_words = [words.index(w) for w in ["gotdepence"] if w in words]
    with CaptionIndex(idx_path, lexicon, documents) as index:
        for doc_id in tqdm.tqdm(range(246923)):
            dic = {}
            count = 1
            postings = index.intervals(doc_id)
            for p in postings:
                if p.end > 300*count:
                    pickle.dump(dic, open('ECJ_doc_stop/Doc_%d_Chunk_%d.p'%(doc_id, count-1),'wb'))
                    dic = {}
                    count += 1
                tokens = index.tokens(0, p.idx, p.len)
                for token in tokens:
                    if token not in stop_words and token not in bs_words:
                        if token in dic:
                            dic[token] += 1
                        else:
                            dic[token] = 1


if __name__ == '__main__':
    main(**vars(get_args()))
