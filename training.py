import numpy as np
import pickle
from scipy.sparse import coo_matrix
from sklearn.decomposition import LatentDirichletAllocation
import glob
import tqdm
import lda

# Load data
print("Starting training...")
#lda = LatentDirichletAllocation(total_samples = 3130377, n_jobs=-1, verbose=10, n_components=15, random_state=0, learning_method="online", max_iter=5, learning_offset=50)
lda = lda.LDA(n_topics = 15, n_iter=150, random_state=0)
row, col, data = np.array(()), np.array(()), np.array(())

matrix_data_list = glob.glob("ECJ_doc/matrix_data_*.p")
np.random.shuffle(matrix_data_list)
for doc in tqdm.tqdm(matrix_data_list[:2]):
    res = pickle.load(open(doc, "rb"))
    row = np.append(row, np.int32(res["I"]))
    col = np.append(col, np.int32(res["J"]))
    data = np.append(data, np.int32(res["data"]))
X = coo_matrix((np.int32(data), (np.int32(row), np.int32(col))))
#print("Partial fitting", doc)
#lda.partial_fit(X)
lda.fit(X)
#    break

print("Training done")
def print_top_words(model, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([str(words[i])
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

print_top_words(lda, 10)
