# -*- coding: utf-8 -*-
"""ml-hw7-ensemble.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C_UCcAWbunFu_1RRSZVqIcu1rBJjdhtx

# **Homework 7 - Bert (Question Answering)**

If you have any questions, feel free to email us at ntu-ml-2023spring-ta@googlegroups.com



Slide:    [Link](https://docs.google.com/presentation/d/15lGUmT8NpLGtoxRllRWCJyQEjhR1Idcei63YHsDckPE/edit#slide=id.g21fff4e9af6_0_13)　Kaggle: [Link](https://www.kaggle.com/competitions/ml2023spring-hw7/host/sandbox-submissions)　Data: [Link](https://drive.google.com/file/d/1YU9KZFhQqW92Lw9nNtuUPg0-8uyxluZ7/view?usp=sharing)

# Prerequisites

## Download Dataset
"""

# from google.colab import drive
# drive.mount('/content/drive')

# # # download link 1
# # # !gdown --id '1TjoBdNlGBhP_J9C66MOY7ILIrydm7ZCS' --output hw7_data.zip

# # # download link 2 (if above link failed)
# # # !gdown --id '1YU9KZFhQqW92Lw9nNtuUPg0-8uyxluZ7' --output hw7_data.zip

# # # download link 3 (if above link failed)
# !gdown --id '1k2BfGrvhk8QRnr9Xvb04oPIKDr1uWFpa' --output hw7_data.zip

# !unzip -o hw7_data.zip

# # For this HW, K80 < P4 < T4 < P100 <= T4(fp16) < V100
# !nvidia-smi

"""## Install packages

Documentation for the toolkit: 
*   https://huggingface.co/transformers/
*   https://huggingface.co/docs/accelerate/index


"""

# # You are allowed to change version of transformers or use other toolkits
# !pip install transformers==4.26.1
# !pip install accelerate==0.16.0

"""# Kaggle (Fine-tuning)

## Task description
- Chinese Extractive Question Answering
  - Input: Paragraph + Question
  - Output: Answer

- Objective: Learn how to fine tune a pretrained model on downstream task using transformers

- Todo
    - Fine tune a pretrained chinese BERT model
    - Change hyperparameters (e.g. doc_stride)
    - Apply linear learning rate decay
    - Try other pretrained models
    - Improve preprocessing
    - Improve postprocessing
- Training tips
    - Automatic mixed precision
    - Gradient accumulation
    - Ensemble

- Estimated training time (tesla t4 with automatic mixed precision enabled)
    - Simple: 8mins
    - Medium: 8mins
    - Strong: 25mins
    - Boss: 2hrs

## Import Packages
"""

import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm.auto import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility
def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
same_seeds(19991124)

"""## Load Model and Tokenizer




 
"""

from transformers import (
  AutoTokenizer,
  AutoModelForQuestionAnswering,
)
pre_model6 = 'saved_model_v6'
pre_model4 = 'saved_model_v4'
pre_model7 = 'saved_model_v7'
pre_model2 = 'saved_model_v2'
pre_model3 = 'saved_model_v3'
# model = AutoModelForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# model = AutoModelForQuestionAnswering.from_pretrained("hfl/chinese-pert-large-mrc").to(device)
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-pert-large-mrc")
model1 = AutoModelForQuestionAnswering.from_pretrained(pre_model3).to(device)
tokenizer = AutoTokenizer.from_pretrained("ianyang66/lert-large_vocab_add")

# You can safely ignore the warning message (it pops up because new prediction heads for QA are initialized randomly)

"""## Read Data

- Training set: 26918 QA pairs
- Dev set: 2863  QA pairs
- Test set: 3524  QA pairs

- {train/dev/test}_questions:	
  - List of dicts with the following keys:
   - id (int)
   - paragraph_id (int)
   - question_text (string)
   - answer_text (string)
   - answer_start (int)
   - answer_end (int)
- {train/dev/test}_paragraphs: 
  - List of strings
  - paragraph_ids in questions correspond to indexs in paragraphs
  - A paragraph may be used by several questions 
"""

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

train_questions, train_paragraphs = read_data("hw7_train.json")
dev_questions, dev_paragraphs = read_data("hw7_dev.json")
test_questions, test_paragraphs = read_data("hw7_test.json")

"""## Tokenize Data"""

# Tokenize questions and paragraphs separately
# 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__ 

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

# You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__ before passing to model

"""## Dataset"""

class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 80
        self.max_paragraph_len = 320
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = 10 #150

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn
        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # A single window is obtained by slicing the portion of paragraph containing the answer
            mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            
            # Change
            if answer_start_token >  self.max_paragraph_len:
               paragraph_start = min(answer_start_token - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len) #不能從0開始
            else:
               paragraph_start = 0
            
            paragraph_end = paragraph_start + self.max_paragraph_len

            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask

train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

"""## Function for Evaluation"""
def evaluate(data, output1):#, output2#, output3):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        # output1.start_logits[k] = output1.start_logits[k] + output2.start_logits[k] + output3.start_logits[k]
        # output1.end_logits[k] = output1.end_logits[k] + output2.end_logits[k] + output3.end_logits[k]
        start_prob, start_index = torch.max(output1.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output1.end_logits[k], dim=0)
        
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        
        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob and start_index <= end_index and end_index - start_index <= 30:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ','')

"""## Training"""
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# from accelerate import Accelerator
# hyperparameters
num_epoch = 1
validation = True
logging_step = 100
learning_rate = 1e-5
train_batch_size = 8

#### TODO: gradient_accumulation (optional)####
# Note: train_batch_size * gradient_accumulation_steps = effective batch size
# If CUDA out of memory, you can make train_batch_size lower and gradient_accumulation_steps upper
# Doc: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
gradient_accumulation_steps = 16

# dataloader
# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
# dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

"""## Testing"""

print("Evaluating Test Set ...")

result = []

model1.eval()
# model2.eval()
# model3.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        output1 = model1(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        result.append(evaluate(data, output1))#, output2, output3))

result_file = "result-tok-v3.csv"
with open(result_file, 'w') as f:	
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
    # Replace commas in answers with empty strings (since csv is separated by comma)
    # Answers in kaggle are processed in the same way
        f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

print(f"Completed! Result is in {result_file}")
