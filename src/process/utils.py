

import csv
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from src.glob_variable import * 

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def write_discharge_summaries(notes_file, out_file):
    tokenizer = RegexpTokenizer(r'\w+')
    #notes_file = '%s/NOTEEVENTS.csv' % (MIMIC_3_DIR)
    with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            print("writing to %s" % (out_file))
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            #header
            next(notereader)
            i = 0
            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]
                    #tokenize, lowercase and remove numerics
                    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text = '"' + ' '.join(tokens) + '"'
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')
                i += 1
    return out_file


def concat_data(labelsfile, notes_file):
    """
        INPUTS:
            labelsfile: sorted by hadm id, contains one label per line
            notes_file: sorted by hadm id, contains one note per line
    """
    with open(labelsfile, 'r') as lf:
        print("CONCATENATING")
        with open(notes_file, 'r') as notesfile:
            outfilename = '%s/notes_labeled.csv' % mimic3_folder
            with open(outfilename, 'w') as outfile:
                w = csv.writer(outfile)
                w.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])

                labels_gen = next_labels(lf)
                notes_gen = next_notes(notesfile)

                for i, (subj_id, text, hadm_id) in enumerate(notes_gen):
                    if i % 10000 == 0:
                        print(str(i) + " done")
                    cur_subj, cur_labels, cur_hadm = next(labels_gen)

                    if cur_hadm == hadm_id:
                        w.writerow([subj_id, str(hadm_id), text, ';'.join(cur_labels)])
                    else:
                        print("couldn't find matching hadm_id. data is probably not sorted correctly")
                        break
    return outfilename



def next_labels(labelsfile):
    """
        Generator for label sets from the label file
    """
    labels_reader = csv.reader(labelsfile)
    #header
    next(labels_reader)

    first_label_line = next(labels_reader)

    cur_subj = int(first_label_line[0])
    cur_hadm = int(first_label_line[1])
    cur_labels = [first_label_line[2]]

    for row in labels_reader:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        code = row[2]
        #keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_labels, cur_hadm
            cur_labels = [code]
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            #add to the labels and move on
            cur_labels.append(code)
    yield cur_subj, cur_labels, cur_hadm

def next_notes(notesfile):
    """
        Generator for notes from the notes file
        This will also concatenate discharge summaries and their addenda, which have the same subject and hadm id
    """
    nr = csv.reader(notesfile)
    #header
    next(nr)

    first_note = next(nr)

    cur_subj = int(first_note[0])
    cur_hadm = int(first_note[1])
    cur_text = first_note[3]
    
    for row in nr:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        text = row[3]
        #keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_text, cur_hadm
            cur_text = text
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            #concatenate to the discharge summary and move on
            cur_text += " " + text
    yield cur_subj, cur_text, cur_hadm



def split_data(labeledfile, base_name):
    print("SPLITTING")
    #create and write headers for train, dev, test
    train_name = '%s_train_split.csv' % (base_name)
    dev_name = '%s_dev_split.csv' % (base_name)
    test_name = '%s_test_split.csv' % (base_name)
    train_file = open(train_name, 'w')
    dev_file = open(dev_name, 'w')
    test_file = open(test_name, 'w')
    train_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    dev_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    test_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")

    hadm_ids = {}

    #read in train, dev, test splits
    for splt in ['train', 'dev', 'test']:
        hadm_ids[splt] = set()
        with open('%s/%s_full_hadm_ids.csv' % (mimic3_folder, splt), 'r') as f:
            for line in f:
                hadm_ids[splt].add(line.rstrip())

    with open(labeledfile, 'r') as lf:
        reader = csv.reader(lf)
        next(reader)
        i = 0
        cur_hadm = 0
        for row in reader:
            #filter text, write to file according to train/dev/test split
            if i % 10000 == 0:
                print(str(i) + " read")

            hadm_id = row[1]

            if hadm_id in hadm_ids['train']:
                train_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['dev']:
                dev_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['test']:
                test_file.write(','.join(row) + "\n")

            i += 1

        train_file.close()
        dev_file.close()
        test_file.close()
    return train_name, dev_name, test_name










