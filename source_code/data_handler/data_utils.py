import os
import random
import numpy as np 
from numpy import ndarray
import pandas as pd 
import scipy.sparse as sp
import torch
import torch.utils.data as data
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from torch.utils.data import default_collate


def custom_collate(batch):
	new_batch = []
	for pos_sample, neg_samples in batch:
		new_batch.append(pos_sample)
		new_batch += neg_samples
	return default_collate(new_batch)


def load_all(config, train_ng_num=99):
	""" We load all the three file here to save time in each epoch. """
	train_dataframe = pd.read_csv(
		config.train_rating, 
		sep='\t', header=None, names=['user', 'item'], 
		usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

	user_num = train_dataframe['user'].max() + 1
	item_num = train_dataframe['item'].max() + 1
	train_dataframe['label'] = 1

	train_data = train_dataframe.values.tolist()

	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)

	val_dataframe = pd.read_csv(
		config.val_rating, 
		sep='\t', header=None, names=['user', 'item'], 
		usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
	val_dataframe['label'] = 1
	val_data = val_dataframe.values.tolist()
	


	test_dataframe_classification = pd.read_csv(
		config.test_rating_classification, 
		sep='\t', header=None, names=['user', 'item', 'label'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
	test_data_classification = test_dataframe_classification.values.tolist()


	test_dataframe_ranking = pd.read_csv(
		config.test_rating_ranking, 
		sep='\t', header=None, names=['user', 'item', 'label'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
	test_data_ranking = test_dataframe_ranking.values.tolist()

	return train_data, test_data_classification, test_data_ranking, user_num, item_num, train_mat, val_data


class CustomDataset(data.Dataset):
	def __init__(self, features, 
				num_user, num_item, train_mat=None, num_ng=0, is_training=None):
		super(CustomDataset, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		self.features_fill = self.features_ps
		self.num_item = num_item
		self.num_user = num_user
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def get_dataframe(self):
		features = self.features_fill if self.is_training and len(self.features_fill) > 0 \
					else self.features_ps
		return pd.DataFrame(features, columns=['user', 'item', 'label'])


	def ng_sample(self, evaluation=False, eval_ng_num=99, eval_full_negatives=False):
		assert self.is_training, 'no need to sampling when testing'

		self.features_ng = []
		self.features_fill = []
		if not evaluation:
			random.shuffle(self.features_ps)
			for x in self.features_ps:
				self.features_fill.append(x)
				u = x[0]
				for _ in range(int(self.num_ng/2)):
					j = np.random.randint(self.num_item)
					while (u, j) in self.train_mat:
						j = np.random.randint(self.num_item)
					self.features_ng.append([u, j, 0])
					self.features_fill.append([u, j, 0])
				i = x[1]
				for _ in range(int(self.num_ng/2)):
					k = np.random.randint(self.num_user)
					while (k, i) in self.train_mat:
						k = np.random.randint(self.num_user)
					self.features_ng.append([k, i, 0])
					self.features_fill.append([k, i, 0])
		else:
			train_df = self.get_dataframe()
			for u, grouped_df in train_df.groupby('user'):
				excluding_job_ids = set(grouped_df['item'])
				for j in excluding_job_ids:
					self.features_fill.append([u, j, 1])
				if eval_full_negatives:
					neg_ids = set([jj for jj in range(self.num_item) if jj not in excluding_job_ids])
				else:
					neg_ids = set()
					pre_neg_ids_len = -1
					while len(neg_ids) < eval_ng_num and len(neg_ids) > pre_neg_ids_len:
						pre_neg_ids_len = len(neg_ids)
						if self.num_item > 10000:
							j_ids = np.random.randint(self.num_item, size=eval_ng_num - len(neg_ids))
							neg_ids = neg_ids.union(set(j_ids)-excluding_job_ids)
						else:
							available_ids = [i for i in range(0, self.num_item) if i not in excluding_job_ids]
							j_ids = random.sample(available_ids, min(eval_ng_num - len(neg_ids), len(available_ids)))
							neg_ids = neg_ids.union(set(j_ids)-excluding_job_ids)
					

				for j in neg_ids:
					self.features_ng.append([u, j, 0])
					self.features_fill.append([u, j, 0])

	def __len__(self):
		features = self.features_fill if self.is_training \
					else self.features_ps
		return len(features)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training \
					else self.features_ps
		user = features[idx][0]
		item = features[idx][1]
		label = features[idx][2]
		return user, item ,label
	
	def change_to_test_dataset(self):
		return self

	def get_non_cluster_dataset(self):
		return self
	
	def get_neg_samples(self, user, item):
		neg_samples = []
		u = user
		for _ in range(int(self.num_ng/2)):
			j = np.random.randint(self.num_item)
			while (u, j) in self.train_mat:
				j = np.random.randint(self.num_item)
			neg_samples.append([u, j, 0])
		i = item
		for _ in range(int(self.num_ng/2)):
			k = np.random.randint(self.num_user)
			while (k, i) in self.train_mat:
				k = np.random.randint(self.num_user)
			neg_samples.append([k, i, 0])
		return neg_samples


class CustomDatasetTrain(CustomDataset):
	def __init__(self, *args, **kwargs):
		super(CustomDatasetTrain, self).__init__(*args, **kwargs)

	def __getitem__(self, idx):
		features = self.features_ps
		user = features[idx][0]
		item = features[idx][1]
		label = features[idx][2]
		pos_sample = [user, item ,label]
		return pos_sample, self.get_neg_samples(user, item)
	
	def change_to_test_dataset(self):
		return CustomDataset(self.features_ps, self.num_user, self.num_item, train_mat=self.train_mat, num_ng=self.num_ng, is_training=self.is_training)

