import pickle
import numpy as np
import scipy
import glob
import tqdm

list_doc = sorted(glob.glob("ECJ_doc_stop/Doc_*_Chunk_*"))
I, J, V = [], [], []
count = 0
chunk = 0
for idx, doc in tqdm.tqdm(enumerate(list_doc)): 
    if count > 500000: 
        pickle.dump({"I": np.array(I), "J": np.array(J), "data": np.array(V)}, open("ECJ_doc_stop/matrix_data_{}.p".format(chunk), "wb") )
        I, J, V = [], [], []
        count = 0
        chunk += 1
    try:
        doc_id = int(doc.split("oc_")[1].split("_Chunk")[0])
        chunk_id = int(doc.split("_Chunk_")[1].split(".p")[0])
        dic = pickle.load(open(doc, "rb"))
        num_words = len(dic.keys())
        I += [doc_id]*num_words
        J += list(dic.keys())
        V += list(dic.values())
        count += num_words
    except Exception:
        continue 




