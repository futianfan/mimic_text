import sys, os, csv, math, operator, logging 
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from time import time
import csv
import os
import gensim.models
from tqdm import tqdm

sys.path.append('./')		#print(os.getcwd())
from src.glob_variable import * 
import utils 

assert current_folder == '/Users/futianfan/Downloads/Gatech_Courses/mimic_text/' 
assert data_folder == '/Users/futianfan/Downloads/Gatech_Courses/mimic_text/data'
assert mimic3_folder == '/Users/futianfan/Downloads/Gatech_Courses/mimic_text/data/mimic3'


'''
STEP 1-5  notes_labeled.csv 

'''

#### 1. combine procedure.csv and diagnosis.csv => ALL_CODES.csv

t1 = time()
proc_file = pd.read_csv('{}/PROCEDURES_ICD.csv'.format(mimic3_folder))
diag_file = pd.read_csv('{}/DIAGNOSES_ICD.csv'.format(mimic3_folder))
proc_file['absolute_code'] = proc_file.apply(lambda row: str(utils.reformat(str(row[4]), True)), axis=1)
diag_file['absolute_code'] = diag_file.apply(lambda row: str(utils.reformat(str(row[4]), True)), axis=1)
allcodes = pd.concat([diag_file, proc_file])
allcodes.to_csv('%s/ALL_CODES.csv' % mimic3_folder, index=False,
               columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'absolute_code'],
               header=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])

df = pd.read_csv('%s/ALL_CODES.csv' % mimic3_folder, dtype={"ICD9_CODE": str})
leng = len(df['ICD9_CODE'].unique()) ### 8994 
print('ALL_CODES has {} different ICD codes'.format(leng))
print('STEP 1. combining procedure.csv and diagnosis.csv takes {} seconds'.format(int(time() - t1)))
### 68 seconds

#### 2. NOTEEVENTS.csv => disch_full.csv 

t1 = time()
disch_full_file = utils.write_discharge_summaries(notes_file ='{}/NOTEEVENTS.csv'.format(mimic3_folder), \
 out_file="{}/disch_full.csv".format(mimic3_folder))
print('STEP 2. write_discharge_summaries takes {} seconds'.format(int(time() - t1)))
###  350 seconds 


#### 3. Compute total num of tokens and num of different tokens

t1 = time()
df = pd.read_csv('%s/disch_full.csv' % mimic3_folder)
df = pd.read_csv('%s/disch_full.csv' % mimic3_folder)
len(df['HADM_ID'].unique())  ####  52726
types = set()
num_tok = 0
for row in df.itertuples():
    for w in row[4].split():
        types.add(w)
        num_tok += 1
print("Num types", len(types))   ### 150854 different words in dictionary 
print("Num tokens", str(num_tok))   ##  79801387 total words
print('STEP 3. Compute total num of tokens  takes {} seconds'.format(int(time() - t1))) 
### 91 seconds 

####  4. sort disch_full && filter ALL_CODES && sort => ALL_CODES_filtered.csv 

t1 = time()
df = pd.read_csv('%s/disch_full.csv' % mimic3_folder)
df = df.sort_values(['SUBJECT_ID', 'HADM_ID'])
dfl = pd.read_csv('%s/ALL_CODES.csv' % mimic3_folder)
dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])
len(df['HADM_ID'].unique()), len(dfl['HADM_ID'].unique())
## Consolidate labels with set of discharge summaries
hadm_ids = set(df['HADM_ID'])  ####  df: disch_full.csv 
with open('%s/ALL_CODES.csv' % mimic3_folder, 'r') as lf:
    with open('%s/ALL_CODES_filtered.csv' % mimic3_folder, 'w') as of:
        w = csv.writer(of)
        w.writerow(['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ADMITTIME', 'DISCHTIME'])
        r = csv.reader(lf)	#header
        next(r) 
        for i,row in enumerate(r):
            hadm_id = int(row[2])
            if hadm_id in hadm_ids:
                w.writerow(row[1:3] + [row[-1], '', '']) 
dfl = pd.read_csv('%s/ALL_CODES_filtered.csv' % mimic3_folder, index_col=None)
len(dfl['HADM_ID'].unique())
dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])
dfl.to_csv('%s/ALL_CODES_filtered.csv' % mimic3_folder, index=False)
df.to_csv('%s/disch_full.csv' % mimic3_folder, index = False)
print('STEP 4. filter ALL_CODES takes {} seconds'.format(int(time() - t1)))
## 47 seconds 


####  5. concatenate 

t1 = time()
df = pd.read_csv('%s/disch_full.csv' % mimic3_folder)
sorted_file = '%s/disch_full.csv' % mimic3_folder
df.to_csv(sorted_file, index=False) 
labeled = utils.concat_data('%s/ALL_CODES_filtered.csv' % mimic3_folder, sorted_file)  ## 52727
### labeled is data/mimic3/notes_labeled.csv => output 
print('STEP 5:concatenate takes {} seconds'.format(int(time() - t1)))

### 63 seconds 

####################################################
#### FLAG notes_labeled.csv 
####################################################

####  6.  compute num of word for notes_labeled.csv

t1 = time()
labeled = '{}/notes_labeled.csv'.format(mimic3_folder)
dfnl = pd.read_csv(labeled)
#Tokens and types
types = set()
num_tok = 0
for row in dfnl.itertuples():
    for w in row[3].split():
        types.add(w)
        num_tok += 1
leng = len(dfnl['HADM_ID'].unique())
print('compute num of word for notes_labeled.csv cost {} seconds '.format(int(time() - t1)))
#### 52 seconds




####  7.  Create train/dev/test splits && Build vocabulary from training data 

t1 = time()
fname = '%s/notes_labeled.csv' % mimic3_folder
base_name = "%s/disch" % mimic3_folder 	#	for output
tr, dv, te = utils.split_data(fname, base_name=base_name)	### output is disch_dev_split.csv, disch_train_split.csv, disch_test_split.csv
vocab_min = 3
vname = '%s/vocab.csv' % mimic3_folder
utils.build_vocab(vocab_min, tr, vname)
## Sort each data split by length for batching
for splt in ['train', 'dev', 'test']:
    filename = '%s/disch_%s_split.csv' % (mimic3_folder, splt)
    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/%s_full.csv' % (mimic3_folder, splt), index=False)
print(' train/dev/test splits && Build vocabulary from training data  cost {} seconds '.format(int(time() - t1)))
#### 205 seconds

####  8.  word embeddings

t1 = time()
## Pre-train word embeddings
w2v_file = utils.word_embeddings('full', '%s/disch_full.csv' % mimic3_folder, 100, 0, 5)				## output is processed_full.w2v  
Y = 'full' #use all available labels in the dataset for prediction
## Write pre-trained word embeddings with new vocab
utils.gensim_to_embeddings('%s/processed_full.w2v' % mimic3_folder, '%s/vocab.csv' % mimic3_folder, Y)		## output is processed_full.embed 
## Pre-process code descriptions using the vocab
utils.vocab_index_descriptions('%s/vocab.csv' % mimic3_folder, '%s/description_vectors.vocab' % mimic3_folder)   ## output is description_vectors.vocab
print(' word embedding cost {} seconds '.format(int(time() - t1)))  ### 12 seconds
################################################################################################
####  9.  Filter each split to the top 50 diagnosis/procedure codes

t1 = time()
## 9.1 find the top-50 codes 
Y = 50
counts = Counter()
dfnl = pd.read_csv('%s/notes_labeled.csv' % mimic3_folder)
for row in dfnl.itertuples():
    for label in str(row[4]).split(';'):
        counts[label] += 1

codes_50 = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
codes_50 = [code[0] for code in codes_50[:Y]]

with open('%s/TOP_%s_CODES.csv' % (mimic3_folder, str(Y)), 'w') as of:
    w = csv.writer(of)
    for code in codes_50:
        w.writerow([code])

## 9.2 filter out some hadm code not in hadm_ids
for splt in ['train', 'dev', 'test']:
    print(splt)
    hadm_ids = set()
    with open('%s/%s_50_hadm_ids.csv' % (mimic3_folder, splt), 'r') as f:
        for line in f:
            hadm_ids.add(line.rstrip())
    with open('%s/notes_labeled.csv' % mimic3_folder, 'r') as f:
        with open('%s/%s_%s.csv' % (mimic3_folder, splt, str(Y)), 'w') as of:
            r = csv.reader(f)
            w = csv.writer(of)
            #header
            w.writerow(next(r))
            i = 0
            for row in r:
                hadm_id = row[1]
                if hadm_id not in hadm_ids:
                    continue
                codes = set(str(row[3]).split(';'))
                filtered_codes = codes.intersection(set(codes_50))
                if len(filtered_codes) > 0:
                    w.writerow(row[:3] + [';'.join(filtered_codes)])
                    i += 1

## 9.3 sort by length
for splt in ['train', 'dev', 'test']:
    filename = '%s/%s_%s.csv' % (mimic3_folder, splt, str(Y))
    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/%s_%s.csv' % (mimic3_folder, splt, str(Y)), index=False)
print(' filter cost {} seconds '.format(int(time() - t1)))  ### 53 seconds










