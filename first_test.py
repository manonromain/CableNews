import numpy as np
import argparse
import os
import time
import traceback
from termcolor import colored, cprint
import pickle
import tqdm

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

    with CaptionIndex(idx_path, lexicon, documents) as index:
        for doc_id in tqdm.tqdm(range(100, 246923)):
            dic = {}
            count = 1
            postings = index.intervals(doc_id)
            for p in postings:
                if p.end > 300*count:
                    pickle.dump(dic, open('ECJ_doc/Doc_%d_Chunk_%d.p'%(doc_id, count-1),'wb'))
                    dic = {}
                    count += 1
                tokens = index.tokens(0, p.idx, p.len)
                for token in tokens:
                    if token in dic:
                        dic[token] += 1
                    else:
                        dic[token] = 1


if __name__ == '__main__':
    main(**vars(get_args()))
