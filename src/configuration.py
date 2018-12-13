import json, os
from glob_variable import current_folder, data_folder, mimic3_folder

def get_config():
	config = {}
	config['embed_file'] = os.path.join(mimic3_folder, 'processed_full.embed')
	config['train_file'] = os.path.join(mimic3_folder, 'train_50.csv')
	config['dev_file'] = os.path.join(mimic3_folder, 'dev_50.csv')
	config['test_file'] = os.path.join(mimic3_folder, 'test_50.csv')
	config['top_50_code_file'] = os.path.join(mimic3_folder, 'TOP_50_CODES.csv')
	config['embed_file'] = os.path.join(mimic3_folder, 'processed_full.embed')

	config['num_class'] = 50 

	config['batch_size'] = 32
	config['max_length'] = 2000 
	config['PAD_CHAR'] = "**PAD**"

	#### CNN 
	config['embed_dim'] = 100 
	config['cnn_out_channel'] = 150 
	config['cnn_kernel_size'] = 10
	config['cnn_stride'] = 1 
	config['cnn_padding'] = 0

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


