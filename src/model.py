import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

#import csv, argparse, os, random, sys, operator
#from time import time
#from tqdm import tqdm
#from collections import defaultdict


from configuration import get_config 
config = get_config()


class CNN_Model(nn.Module):

	def __init__(self, **config):
		super(CNN_Model, self).__init__()

		self.embed_dim = config['embed_dim']  ### in_channels 
		self.cnn_out_channel = config['cnn_out_channel']
		self.cnn_kernel_size = config['cnn_kernel_size']
		self.cnn_stride = config['cnn_stride']
		self.cnn_padding = config['cnn_padding']
		self.num_class = config['num_class']

		self.conv = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.cnn_out_channel, kernel_size=self.cnn_kernel_size,\
			 stride=self.cnn_stride, padding=self.cnn_padding)
		xavier_uniform_(self.conv.weight)
		#self.maxpool = nn.MaxPool1d(kernel_size = )

		self.fc = nn.Linear(self.cnn_out_channel, self.num_class)
		xavier_uniform_(self.fc.weight)

	@staticmethod
	def _get_loss(xout, y):
		return F.binary_cross_entropy(xout, y)

	def forward(self, X, y):
		"""
			input: batch_size, max_len, embed_dim 
		"""
		y = Variable(torch.FloatTensor(y))
		X = Variable(X)			### batch_size, max_len, embed_dim
		X = X.transpose(2,1)		### batch_size, embed_dim (in_channel), max_len
		Xconv = self.conv(X)	### batch_size, out_channels, new_len  
		Xmaxpool = F.max_pool1d(torch.tanh(Xconv), kernel_size=Xconv.size()[2])	### 32, 150, 1
		Xmaxpool = Xmaxpool.squeeze(2)
		Xout = torch.sigmoid(self.fc(Xmaxpool))  ### batch_size, num_class
		return self._get_loss(Xout, y), Xout, y 
		#return Xout


if __name__ == '__main__':
	from configuration import get_config
	config = get_config()
	cnn = CNN_Model(**config)

	from data_read import * 
	code2idx, idx2code = get_top_50_code(config['top_50_code_file'])  ### 1. 
	word2idx, idx2word, embed_mat = load_embedding(config['embed_file'])  ### 2.
	embed_mat = torch.FloatTensor(embed_mat) 
	embedding = nn.Embedding.from_pretrained(embed_mat)
	trainData = CSV_Data_Reader(batch_size = config['batch_size'], filename = config['train_file'], max_length = config['max_length'], \
		  code2idx = code2idx, word2idx = word2idx)
	for i in range(1):
		batch_embed, batch_label = trainData.next(embedding)
		#print(type(batch_embed))
		output = cnn(batch_embed, batch_label)
		if i % 100 == 0: print(batch_embed.shape)









