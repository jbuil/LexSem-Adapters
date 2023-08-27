# -*- coding: utf-8 -*-
"""
Script for lexical relation classification with adapters
Adapted from the code originally used in paper: 
	"No clues, good clues: Out of context Lexical Relation Classification"
	 
Script usage example:

$python scripts/lrc_train_evaluate.py \
	--train_templates "' <W1> ' <SEP> ' <W2> '"   \
	--test_templates "' <W1> ' <SEP> ' <W2> '"   \
	--model  "roberta-base" \
  	--use_adapters True \
	--nepochs 10 \
	--dir_output_results "results/" \
	--batch_size 32 \
	--warm_up 0.1 \
	--nrepetitions 1 \
	--dataset "EVALution" \
	--date `date "+%D-%H:%M:%S"` \
	--train_file "lexical_datasets/EVALution/train.tsv" \
	--test_file "lexical_datasets/EVALution/test.tsv" \
	--val_file "lexical_datasets/EVALution/val.tsv" # comment this line, if there is not val dataset
"""

import numpy as np
import pandas as pd
import re
import os
import torch
from torch import nn
from random import randint
import argparse
from datetime import datetime
import logging

from sklearn.metrics import top_k_accuracy_score,confusion_matrix, classification_report
from scipy.stats import entropy

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoAdapterModel, AdapterTrainer
from datasets import Dataset, load_metric, load_dataset

parser = argparse.ArgumentParser(description='Train and test models to classify relations.')
parser.add_argument("-ftrain", "--train_file", required=True, help="Path to tab sep text test file: two words and a relation name by line")
parser.add_argument("-fval","--val_file", required=False, help="Path to tab sep text  val file: two words and a relation name by line")
parser.add_argument("-ftest", "--test_file", required=True, help="Path to tab sep text test file: two words and a relation name by line")
parser.add_argument("-ttrain", "--train_templates", required=True, nargs='+', help="List of templates to verbalize two words for train: They should contain <W1> and <W2> to substitute words in a line.")
parser.add_argument("-ttest", "--test_templates", required=False, nargs='+', help="List of templates to verbalize two words for test: They should contain <W1> and <W2> to substitute words in a line.")
parser.add_argument("-m", "--model", required=True, help="Model name checkpoint")
parser.add_argument("-e", "--nepochs", required=True, type=int, help="Number training epochs")
parser.add_argument("-o", "--dir_output_results", default="./", help="Directory to save the test results")
parser.add_argument("-rep", "--nrepetitions", default=1, type=int, help="Number of times the experiment is run")
parser.add_argument("-b", "--batch_size", required=True, type=int, help="Batch size")
parser.add_argument("-wup", "--warm_up", required=False, type=float, default=0.0, help="Warm up ratio for training")
parser.add_argument("-data", "--dataset", required=True, help="Name of the dataset for fine-tuning")
parser.add_argument("-params", "--parameters_list", required=False, help="")
parser.add_argument("-d", "--date", required=False, help="Experiment date")
parser.add_argument("-raw", "--raw_model", default=False, type=bool, help="If True, it is used a no trained model. Default: False")
parser.add_argument("-adapters", "--use_adapters", default=False, type=bool, help="If True, adapters are added to all layers of the model. Default: False")

#parameters
args = parser.parse_args()
model_name = args.model
train_templates = args.train_templates
test_templates = args.test_templates
if test_templates == None:
    test_templates = train_templates
train_file = args.train_file
test_file = args.test_file
val_file = args.val_file #None
total_repetitions = args.nrepetitions
batch_size = args.batch_size
warm_up = args.warm_up
name_dataset = args.dataset
params = args.parameters_list
date = args.date
output = args.dir_output_results
total_epochs = args.nepochs
warmup_r = args.warm_up
is_raw = args.raw_model
use_adapters = args.use_adapters

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

exc_message = "Train templates and test templates must be lists of equal size.\nTrain template list contains {:d} templates and test template list contains {:d}"
if len(train_templates) != len(test_templates):
    raise Exception(exc_message.format(len(train_templates), len(test_templates)))
				  
datasets = ['evalution', 'cogalexv', 'root9']

# tokenizer for model
tokenizer = AutoTokenizer.from_pretrained(model_name)

d = name_dataset.lower()
if not d in datasets:
    logging.warning("Parameter --dataset is not one of: " + ', '.join(datasets))

task_name = 'LRC_' + name_dataset

verb_dict = None

if name_dataset.lower() not in train_file.lower().split("/"):
    print("It seems that --dataset (" + name_dataset + ") does not correspond with --train_file (" + train_file + ")")
    print("Are you sure to continue[y/n]?")
    follow = input()
    if follow != 'y':
        print("Finished")
        raise SystemExit
                         
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#create output dir, if it does not exist
try:
    os.makedirs(output)
except:
   pass 

def verb_row(row, template, tokenizer, verb_dict=None):
    """
    Create a verbalization of a row (a pair of words and
    a relation label) following a template that can contains 
    <W1>, <W2>, <LABEL> and <SEP> to substitute source, 
    target words, the relation label and the special token SEP
    of a tokenizer. If verb_dict is not None, verb_dict is a 
    dictionary that must contains pairs (key, value)
    where key is a relation label, and value is the verbalization
    of the relation label uses to sustitute <LABEL> in the template.
    
    Args:
      row -- a series with 'source', 'target' and 'rel'
        template -- a string with (possible) <W1>, <W2>, <LABEL> and <SEP> 
      tokenizer -- a tokenizer with its special tokens
      verb_dict -- dictionary with the verbalizations (values) of 
        the relation labels (keys)
    
    Returns:
      a dictionary, {'verb':verbalization}, with the key 'verb'
      and the verbalization of the row following the template.
    """
    w1 = str(row['source'])
    w2 = str(row['target'])
    lab = str(row['rel']).lower()
    sentence = re.sub("<W1>", w1, template)
    sentence = re.sub("<W2>", w2, sentence)
    sentence = re.sub("<SEP>", tokenizer.sep_token, sentence)
    if verb_dict != None:
        if lab in verb_dict:
            lab = verb_dict[lab].strip()
        sentence = re.sub("<LABEL>", lab, sentence)
    return {'verb':sentence}

def preprocess_function(rows, tokenizer):
    """ tokenize the column 'verb' of the rows"""
    inputs = tokenizer(rows['verb'], truncation=True, padding='max_length', max_length=64)
    return inputs

def compute_metrics(eval_pred):
    '''
    Compute metrics for a Trainer.

    Args:
     eval_pred: object of type transformers.EvalPrediction. It is a tuple with 
     predictions (logits) and real labels.

    Returns:
     A dictionary of metrics {'name_metric1':value1,...}
    '''
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    return metric.compute(predictions=predictions, references=labels, average='macro')    

def results_row(row, tokenizer):
    pred = (row['pred_label'])
    gold = (row['real_label'])
    if pred == gold:
      row['results'] = True
    else:
      row['results'] = False
	
    toks_s = tokenizer.tokenize(" " + row['source'])
    toks_t = tokenizer.tokenize(" " + row['target'])
    row['toks_source'] = str(toks_s)
    row['toks_target'] = str(toks_t)
    row['n_toks_source'] = len(toks_s)
    row['n_toks_target'] = len(toks_t)
    return (row)

msgFinetuning = '''Starting fine-tuning with: 
  - model: {:s}
  - train file: {:s} 
  - test file: {:s}
  - val file: {:s}
  - train templates: {:s}
  - test templates: {:s}
*****************************************'''
logging.info(msgFinetuning.format(model_name, train_file, test_file, 
           val_file if val_file != None else "None", 
           str(train_templates), str(test_templates)))

# PREPARE DATA
# load train/test files to datasets dict. Also load val file, if it exists
# datasets contains lines with three strings: source_word, target_word, rel_label
data_files = {'train':train_file,'test':test_file}
if val_file != None:
	data_files['val'] = val_file
all_data = load_dataset('csv', data_files=data_files, sep='\t', header=None, names=['source', 'target', 'labels'], keep_default_na=False)

# create the column 'rel', copy of column 'labels'
all_data = all_data.map(lambda x: {'rel':x['labels']})

# trasform column 'labels' to a integer with a label id. Needed for the tokenizer
all_data = all_data.class_encode_column('labels')
print(all_data)

# load metric
metric_name = 'f1'
metric = load_metric(metric_name)

# seeds to avoid equal trainings
seeds = [randint(1,100) for n in range(total_repetitions)]
while len(set(seeds)) != total_repetitions:
	seeds = [randint(1,100) for n in range(total_repetitions)]
	
print(seeds)

for train_template, test_template in zip(train_templates, test_templates):
    for i in range(total_repetitions):
        print("****** Repetition: " + str(i+1) + "/" + str(total_repetitions))
        NUM_LABELS = all_data['train'].features['labels'].num_classes

        if use_adapters:
          model = AutoAdapterModel.from_pretrained(model_name)
          config = model.config
          if is_raw:
            print('Using LM raw model...')
            model = AutoAdapterModel.from_config(config=config)
          model.add_classification_head(task_name, num_labels=NUM_LABELS)
          model.add_adapter(task_name)
          model.train_adapter(task_name)
        else: 
          # No adapters, just the model
          model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
          config = model.config
          if is_raw:
            print('Using LM raw model...')
            model = AutoModelForSequenceClassification.from_config(config=config)
        
        # verbalize the datasets with template
        all_data['train'] = all_data['train'].map(verb_row, fn_kwargs={'tokenizer':tokenizer, 'template':train_template, 'verb_dict':verb_dict})
        all_data['test'] = all_data['test'].map(verb_row, fn_kwargs={'tokenizer':tokenizer, 'template':test_template, 'verb_dict':verb_dict})
        if val_file != None:
            all_data['val'] = all_data['val'].map(verb_row, fn_kwargs={'tokenizer':tokenizer, 'template':test_template, 'verb_dict':verb_dict})   
        print(all_data['train']['verb'][0:10])        
        print(all_data['test']['verb'][0:10])
        
		    # encode data for language model
        encoded_all_data = all_data.map(preprocess_function, batched=True, batch_size=None, fn_kwargs={'tokenizer':tokenizer})
		
		    # separate the splits in datasets dict
        encoded_verb_train = encoded_all_data['train']
        if val_file != None:
            encoded_verb_val = encoded_all_data['val']
        encoded_verb_test = encoded_all_data['test']
		
        encoded_verb_train.set_format("torch")
        if val_file != None:
            encoded_verb_val.set_format("torch")
        encoded_verb_test.set_format("torch") 
        
        args_train = TrainingArguments(
          output_dir='my_checkpoints',
          overwrite_output_dir=True,
          evaluation_strategy="epoch" if val_file != None else "no",
          save_strategy="epoch" if val_file != None else "no",
          per_device_train_batch_size=batch_size,
          per_device_eval_batch_size=batch_size*2,
          optim="adamw_torch",
          learning_rate=1e-4,
          #weight_decay=0.01,
          warmup_ratio=warmup_r,
          #fp16=True,
          logging_steps=10,
          load_best_model_at_end=True if val_file != None else False,
          metric_for_best_model=metric_name,
          num_train_epochs=total_epochs,
          report_to='all',
          seed=seeds[i],
			    save_total_limit = 1
		    )    

        if use_adapters:
          trainer = AdapterTrainer(
            model, #model to train
            args_train,  #arguments to train
            train_dataset=encoded_verb_train,
            eval_dataset = encoded_verb_val if val_file != None else None,
            tokenizer=tokenizer, #it is needed the tokenizer that encoded the data for batch
            compute_metrics=compute_metrics, #to compute metric of the model,
          )
        else:    
          trainer = Trainer(
            model, #model to train
            args_train,  #arguments to train
            train_dataset=encoded_verb_train,
            eval_dataset = encoded_verb_val if val_file != None else None,
            tokenizer=tokenizer, #it is needed the tokenizer that encoded the data for batch
            compute_metrics=compute_metrics, #to compute metric of the model,
          )

		    #start training
        trainer.train()
		
        #predict test
        predicciones = trainer.predict(test_dataset=encoded_verb_test)
        pred = np.argmax(predicciones.predictions, axis = 1)
        
        real_rel_test = encoded_verb_test.features['labels'].int2str(encoded_verb_test['labels'])
        pred_rel_test = encoded_verb_test.features['labels'].int2str(pred)
        results_acc = (classification_report(real_rel_test, pred_rel_test, digits=3, output_dict=True))
        print(results_acc)
        encoded_verb_test.set_format('numpy')
        results_words = pd.DataFrame({'pred_label':pred, 'pred_rel':pred_rel_test, 'real_label':predicciones.label_ids, 'real_rel':real_rel_test, 'source':encoded_verb_test['source'], 'target':encoded_verb_test['target']})
        results_words = results_words.apply(results_row, axis=1, tokenizer=tokenizer)
    		
        sfmax = nn.Softmax(dim=1)
        probs = sfmax(torch.tensor(predicciones.predictions))
        probs_df = pd.DataFrame(probs.numpy(), columns=encoded_verb_test.features['labels'].names)
        chaos = entropy(probs, axis = 1, base = 2)
        chaos_df =  pd.DataFrame(chaos, columns=['entropy'])

        results_words = pd.concat([results_words, probs_df, chaos_df], axis = 1)

        now = datetime.now()
        now = now.strftime('%y-%m-%d_%H-%M-%S')  
        fname = output + name_dataset + '_I' + str(i).zfill(2) + "_" + now
        with open((fname + '.txt') , 'w') as f:
            print(vars(args), file=f)
            print(date, file=f)
            print(results_acc, file=f)
			
            results_words.to_csv(fname + '.csv', index=False)         
