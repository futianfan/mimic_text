import json, os
from glob_variable import current_folder, data_folder, mimic3_folder

def get_config():
	config = {}
	config['embed_file'] = os.path.join(mimic3_folder, 'processed_full.embed')
	config['train_file'] = os.path.join(mimic3_folder, 'train_50.csv')
	config['dev_file'] = os.path.join(mimic3_folder, 'dev_50.csv')
	config['test_file'] = os.path.join(mimic3_folder, 'test_50.csv')
	

	'''
	config[''] = 
	'''


	return config

'''
def write_config(filename = './config'):
	config = {}

	config['batch_size'] = 16 

	config['LSTM_hidden_size'] = 50 

	config['CNN_hidden_size'] = 50 

	json.dump(config, open(filename, 'w'))
'''
#if __name__ == '__main__':
	
'''	fname = './config'
	write_config()
	conf = json.load(open(fname, 'r'))
	assert conf['batch_size'] == 16
'''


