import json, os

def get_config():
	config = {}
	current_folder = '/Users/futianfan/Downloads/Gatech_Courses/mimic_text/' 
	data_folder = os.path.join(current_folder, 'data')
	mimic3_folder = os.path.join(data_folder, 'mimic3')

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
	assert config['embed_dim'] == 100 
	config['cnn_out_channel'] = 250 
	config['cnn_kernel_size'] = 10
	config['cnn_stride'] = 1 
	config['cnn_padding'] = 0


	#### train
	config['learning_rate'] = 1e-2

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


