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
import argparse

def main(folder):
    word2idx = pickle.load(open(os.path.join(folder, "word_idx.p"), "rb"))
    print(word2idx)
    # Load seed topics
    seed_topics_dic, topics = seed_topics(word2idx)
    import pdb; pdb.set_trace()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train the graph network.')
    parser.add_argument('folder', help='Training data folder')
    args = parser.parse_args()
    main(args.folder)
