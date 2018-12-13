import numpy as np
import csv, os
import torch 
from torch import nn 
from data_read import * 
from model import CNN_Model

def train():
	from configuration import get_config
	config = get_config()
	cnn = CNN_Model(**config)

	code2idx, idx2code = get_top_50_code(config['top_50_code_file'])  ### 1. 
	word2idx, idx2word, embed_mat = load_embedding(config['embed_file'])  ### 2.
	embed_mat = torch.FloatTensor(embed_mat) 
	embedding = nn.Embedding.from_pretrained(embed_mat)
	trainData = CSV_Data_Reader(batch_size = config['batch_size'], filename = config['train_file'], max_length = config['max_length'], \
		  code2idx = code2idx, word2idx = word2idx)
	opt_  = torch.optim.SGD(cnn.parameters(), lr=config['learning_rate'])
	for i in range(1000):
		batch_embed, batch_label = trainData.next(embedding)
		loss = cnn(batch_embed, batch_label)
		opt_.zero_grad()
		loss.backward() 
		opt_.step() 
		loss_value = loss.data[0]
		print('iteration {}, loss value: {}'.format(i, loss_value))

if __name__ == '__main__':
	train()


