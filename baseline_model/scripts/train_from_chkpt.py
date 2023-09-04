# %% [markdown]
# Used YouTube video as reference
# 
# https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb
# 
# https://github.com/huggingface/transformers/tree/main/examples/legacy/seq2seq
# 
# https://youtu.be/T2fISIRogkg

# %% [markdown]
# ***
# ### Imports and Globals
# ***

# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric

import nltk
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


# %%
# Important for saving model
nltk.download('punkt')

# %%
RANDOM_SEED = 99

DATA_FILES = 'casino' # | 'casino_w_task_data'
EPOCHS = 30

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = 'model_saves'
MAX_INPUT_LEN = 1024  # Max length sequence in baseline_casino.csv is 824
MAX_OUTPUT_LEN = 192
MODEL_NAME = "t5-base"
# MODEL_NAME = "t5-small"

# %% [markdown]
# ***
# ### Creating the Model/Tokenizer
# ***

# %%
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_INPUT_LEN, eos_token='<EOS>')
special_tokens = {'additional_special_tokens': ['<CONTEXT>', '<HISTORY>', '<YOU>', '<THEM>']}
tokenizer.add_special_tokens(special_tokens)

model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id=tokenizer.eos_token_id

# Can use following instead for better generalization when testing other architectures
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_INPUT_LEN)

# %% [markdown]
# ***
# ### Data Processing
# ***

# %%
# Construct Dataset
def preprocess_function(examples):
    # Tokenize inputs/outputs
    model_inputs = tokenizer(examples['input_seq'], max_length=MAX_INPUT_LEN, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['response'], max_length=MAX_OUTPUT_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# Loading Dataset
b_url = f'data/splits/{DATA_FILES}/'
raw_datasets = load_dataset("csv", data_files={'train': f'{b_url}train.csv', 'eval': f'{b_url}eval.csv', 'test': f'{b_url}test.csv'})
raw_datasets = raw_datasets.shuffle(seed=RANDOM_SEED)

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# %%
raw_datasets

# %% [markdown]
# ***
# ### Training
# ***

# %%
batch_size = 8
model_name = MODEL_NAME.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{MODEL_DIR}/{model_name}-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    # fp16=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

bertscore = load_metric('bertscore')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

    bleu_result = []
    for lab, pred in zip(decoded_labels, decoded_preds):
        # Syntax: sentence_blue(reference, candidate AKA prediction, weights) 
        sample_score = sentence_bleu([lab.lower().split()], pred.lower().split(), weights=(0.50, 0.25, 0.15, 0.10))
        bleu_result.append(sample_score)
    bleu_result = np.mean(bleu_result)

    return {'bleu': bleu_result, 'bertscore_precision': np.mean(bertscore_result['precision']), 'bertscore_recall': np.mean(bertscore_result['recall']), 'bertscore_f1': np.mean(bertscore_result['f1'])}

# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model('final_model')
