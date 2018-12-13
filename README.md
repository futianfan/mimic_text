
baseline MIMIC text ICD-code 

the data-processing part rely heavily on 
	[CAML-MIMIC](https://github.com/jamesmullenbach/caml-mimic)

original input:

data

|   D_ICD_DIAGNOSES.csv

|   D_ICD_PROCEDURES.csv

└───mimic3/

|   |   NOTEEVENTS.csv

|   |   DIAGNOSES_ICD.csv

|   |   PROCEDURES_ICD.csv

|   |   *_hadm_ids.csv


====source code====
src/
	data_reader.py
	model.py
	configuration.py
	train.py

	config -> data -> model -> train -> 

====How to run=====
./run.sh


