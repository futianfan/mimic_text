import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

#import csv, argparse, os, random, sys, operator
#from time import time
from tqdm import tqdm
#from collections import defaultdict


from configuration import get_config 
config = get_config()


class BaseModel(nn.Module):

	def __init__(self, **config):
		super(BaseModel, self).__init__()
		self.embed_file = config['embed_file']


	






