import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

from random import randint

import numpy as np
import torch

# Load model
from InferSent.models import InferSent

import pickle
import nltk.data

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('channel', type=str,
						help='Directory containing index files')
	return parser.parse_args()

Ticks = np.array(['Other', 'Politics', 'Media', 'Fashion', 'Foreign Policy', 'Immigration', 
		 'Economy', 'Health', 'Art', 'Gender', 'Sport', 'Violence', 'Climate'])

class classifer():
	def __init__(self):
		self.lr = 5 * 1e-4
		self.n_classes = 13
		
		self.make_placeholders()
		self.make_nn()
		self.make_loss()
		self.make_train_op()
		
		self.sess = tf.Session()
		self.sess.run(tf.initializers.global_variables())
		
		self.saver = tf.train.Saver()
		
	def save(self):
		self.saver.save(self.sess, 'nn-classifier-v2')
		
	def load(self, name):
		self.saver.restore(self.sess, name)
		
	def make_placeholders(self):
		self.input = tf.placeholder(tf.float32, shape=[None, 4096], name='X')
		self.label = tf.placeholder(tf.int32, shape=[None, self.n_classes], name='label')
		
	def make_nn(self):
		X = tf.layers.dense(self.input, 512, activation=tf.nn.relu,
							  kernel_initializer=tf.keras.initializers.glorot_normal(), name='Dense_1')
		X = tf.layers.dense(X, 512, activation=tf.nn.relu,
							  kernel_initializer=tf.keras.initializers.glorot_normal(), name='Dense_2')
		X = tf.layers.dense(X, 512, activation=tf.nn.relu,
							  kernel_initializer=tf.keras.initializers.glorot_normal(), name='Dense_3')
		X = tf.layers.dense(X, 512, activation=tf.nn.relu,
							  kernel_initializer=tf.keras.initializers.glorot_normal(), name='Dense_4')
		self.logit = tf.layers.dense(X, self.n_classes, activation=None,
							  kernel_initializer=tf.keras.initializers.glorot_normal(), name='logits')
		
		self.prediction = tf.nn.softmax(self.logit)
		
	def make_loss(self):
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.label, self.logit))
	
	def make_train_op(self):
		self.optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_op = self.optimizer.minimize(self.loss)
	
	def train(self, X, Y):
		loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.input:X, self.label:Y})
		return loss
	
	def predict(self, X):
		prediction = self.sess.run([self.prediction], feed_dict={self.input:X})
		return prediction

def process(channel):
	# Load the Classifier
	tf.reset_default_graph()
	NN = classifer()
	NN.load('nn-classifier-v2')

	# Load the sentence embedder
	model_version = 1
	MODEL_PATH = "InferSent/encoder/infersent%s.pkl" % model_version
	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
				'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
	model = InferSent(params_model)
	model.load_state_dict(torch.load(MODEL_PATH))
	# Keep it on CPU or put it on GPU
	use_cuda = False
	model = model.cuda() if use_cuda else model

	# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
	W2V_PATH = 'InferSent/GloVe/glove.840B.300d.txt' if model_version == 1 else 'InferSent/fastText/crawl-300d-2M.vec'
	model.set_w2v_path(W2V_PATH)
	# Load embeddings of K most frequent words
	model.build_vocab_k_words(K=100000)

	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	all_files = glob.glob('../files/CableNews/%s/*.p'%channel)
	read_files = pickle.load(open('%s_visit.p'%(channel), 'rb'))
	counter = len(read_files)

	for file in tqdm(all_files):
		if file in read_files:
			continue
		else:
			read_files.append(file)
			if np.random.rand() < 0.3:
				pickle.dump(read_files, open('%s_visit.p'%(channel), 'wb'))     
				
		res = pickle.load(open(file, 'rb'))
		results = {}
		prev_text = ""
		all_text = []
		all_keys = []
		for key in res.keys():
			meta_data = res[key][0] # First in the list
			if len(meta_data['text']) < 10:
				continue

			# Make sure we drop the duplicates: Texts should be differents
			current_text = meta_data['text'][:10]
			if current_text == prev_text:
				continue
			else:
				prev_text = current_text
			
			text = tokenizer.tokenize(meta_data['text'])
			if len(text) <= 2:
				continue
			# Drop the first sentence
			text = text[1:]
			senteces = []
			for s in text: #Drop super small and super large senteces
				if len(s.split()) > 30 and len(s.split()) < 50:
					senteces.append(s)
			if len(senteces) == 0:
				continue
			# Calculate the embedding
			all_text.extend(senteces)
			all_keys.extend([key]*len(senteces))
		if len(all_text) == 0:
			continue
		all_embed = model.encode(all_text, bsize=128, tokenize=True, verbose=False)
		all_predictions = NN.predict(all_embed)[0] # Merge the probabilties and take top 2:
		prev_key = None
		total_prob = np.zeros((13, 1))
		key_counter = 0
		for current_key in all_keys:
			if current_key==prev_key:
				total_prob[:, 0] += all_predictions[key_counter, :]
			else:
				Topics = Ticks[np.flip(np.argsort(total_prob[:, 0])[-2:])]; 
				Probs = np.flip(np.sort(total_prob[:, 0])[-2:]) * 100
				results[current_key] = {'Topics': list(Topics), 'Probs': list(Probs), 'gender': res[current_key][0]['gender'],
						   'persons': res[current_key][0]['persons'], 'locations': res[current_key][0]['locations']}
				prev_key = current_key
				total_prob = np.zeros((13, 1))
				total_prob[:, 0] += all_predictions[key_counter, :]
			key_counter += 1
		pickle.dump(results, open('processed_data/%s/%d.p'%(channel, counter), 'wb'))
		counter += 1

if __name__ == '__main__':
	process(**vars(get_args()))