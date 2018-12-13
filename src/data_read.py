import csv, os
import numpy as np 
import torch 
from torch import nn 
def get_top_50_code(top_50_code_file):
	with open(top_50_code_file, 'r') as fin:
		r = csv.reader(fin)
		top_50_code = [line[0] for line in r]
		code2idx = {code:idx for idx, code in enumerate(top_50_code)}
		idx2code = {idx:code for idx, code in enumerate(top_50_code)}
	return code2idx, idx2code

def load_embedding(embed_file):
	with open(embed_file, 'r') as fin:
		lines = fin.readlines()
		embed_dim = len(lines[0].split()) - 1
		assert embed_dim == 100 
		word_lst = list(map(lambda line:line.split()[0], lines))
		word2idx = {word:idx for idx,word in enumerate(word_lst)}
		idx2word = {idx:word for idx,word in enumerate(word_lst)}
		embed_mat = np.zeros((len(lines), embed_dim), dtype = np.float)
		for idx, line in enumerate(lines):
			vec = line.strip().split()[1:]
			vec = np.array([float(i) for i in vec])
			embed_mat[idx,:] = vec
	return word2idx, idx2word, embed_mat


'''
def code2idx_func(code, code2idx):
	code = code.strip().split(';')
	code_idx = [code2idx[i] for i in code]
	return code_idx

def text2idx_func(text, word2idx, max_length):
	text = text.strip().split()[:max_length]
	word_idx = [word2idx[word] for word in text]
	return word_idx
'''

class CSV_Data_Reader:
	def __init__(self, batch_size, filename, max_length, code2idx, word2idx):
		with open(filename, 'r') as fin:
			lines = fin.readlines()[1:]
			self.total_num = len(lines)
			self.batch_size = batch_size
			self.num_of_iter_in_epoch = int(np.ceil(self.total_num / self.batch_size))
			self.max_length = max_length
			
			self.batch_id = 0 
			self.random_idx = np.arange(self.total_num)
			np.random.shuffle(self.random_idx)

			def code2idx_func(code):
				code = code.strip().split(';')
				code_idx = [code2idx[i] for i in code]
				return code_idx

			def text2idx_func(text):
				text = text.strip().split()[:max_length]
				word_idx = [word2idx[word] if word in word2idx else 0 for word in text]
				return word_idx

			self.code = list(map(code2idx_func, [line.split(',')[3] for line in lines]))
			self.text = list(map(text2idx_func, [line.split(',')[2] for line in lines]))


	def next(self, embedding):
		bgn_num, end_num = self.batch_id*self.batch_size, self.batch_id*self.batch_size + self.batch_size
		indx = self.random_idx[bgn_num:end_num]
		
		### reshuffle 
		self.batch_id += 1
		if self.batch_id == self.num_of_iter_in_epoch:
			self.batch_id = 0
			np.random.shuffle(self.random_idx)
		### reshuffle 

		text_seq = [self.text[i] for i in indx]
		max_len = max([len(tex) for tex in text_seq])
		for tex in text_seq:
			tex.extend([0] * (max_len - len(tex)))
		### zero-padding
		text_seq = torch.LongTensor(text_seq)
		text_embedding = embedding(text_seq)
		return text_embedding, [self.code[i] for i in indx]

	### to do 1. data => embedding + padding: word => idx + padding => embedding 
	### to do 2. label => idx 



if __name__ == '__main__':
	from configuration import get_config
	config = get_config()
	code2idx, idx2code = get_top_50_code(config['top_50_code_file'])  ### 1. 
	word2idx, idx2word, embed_mat = load_embedding(config['embed_file'])  ### 2.
	embed_mat = torch.FloatTensor(embed_mat) 
	embedding = nn.Embedding.from_pretrained(embed_mat)

	
	trainData = CSV_Data_Reader(batch_size = config['batch_size'], filename = config['train_file'], max_length = config['max_length'], \
		  code2idx = code2idx, word2idx = word2idx)
	for i in range(1):
		batch_embed, batch_label = trainData.next(embedding)
		print(type(batch_embed))
		if i % 100 == 0: print(batch_embed.shape)
	pass






