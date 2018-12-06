import json

def write_config(filename = './config'):
	config = {}

	config['batch_size'] = 16 

	config['LSTM_hidden_size'] = 50 

	config['CNN_hidden_size'] = 50 

	json.dump(config, open(filename, 'w'))

if __name__ == '__main__':
	fname = './config'
	write_config()
	conf = json.load(open(fname, 'r'))
	assert conf['batch_size'] == 16

