import numpy as np
import pickle
from scipy.sparse import coo_matrix
from sklearn.decomposition import LatentDirichletAllocation
import glob
import tqdm
import lda
from utils import get_lexicon

words = get_lexicon()
# Load data
print("Starting training...")
#lda = LatentDirichletAllocation(total_samples = 3130377, n_jobs=-1, verbose=10, n_components=15, random_state=0, learning_method="online", max_iter=5, learning_offset=50)
lda = lda.LDA(n_topics = 5, n_iter=150, random_state=0)
row, col, data = np.array(()), np.array(()), np.array(())

matrix_data_list = ["ECJ_doc_stop/matrix_data_514.p"] #, "ECJ_doc_stop/matrix_data_199.p"]#glob.glob("ECJ_doc_stop/matrix_data_*.p")
np.random.shuffle(matrix_data_list)
for doc in tqdm.tqdm(matrix_data_list[:1]):
    print("Partial fitting", doc)
    res = pickle.load(open(doc, "rb"))
    row = np.append(row, np.int32(res["I"]))
    col = np.append(col, np.int32(res["J"]))
    data = np.append(data, np.int32(res["data"]))

#print("BS", 644928 in col)
X = coo_matrix((np.int32(1000*data), (np.int32(row), np.int32(col))))
print(X.tocsr()[:, 644928].todense().any())
#exit(0)
#lda.partial_fit(X)
lda.fit(X)
#    break
print(lda.components_.shape)
print(lda.components_)
print("Training done")
def print_top_words(model, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        print(topic)
        message += " ".join([str(words[i])
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

print_top_words(lda, 10)

