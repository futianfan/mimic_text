import sys, os

current_folder = '/Users/futianfan/Downloads/Gatech_Courses/mimic_text/' 
data_folder = os.path.join(current_folder, 'data')
mimic3_folder = os.path.join(data_folder, 'mimic3')

PAD_CHAR = "**PAD**"

trainFile = os.path.join(mimic3_folder, 'train_50.csv')
devFile = os.path.join(mimic3_folder, 'dev_50.csv')
testFile = os.path.join(mimic3_folder, 'test_50.csv')
top_50_code_file = os.path.join(mimic3_folder, 'TOP_50_CODES.csv')



