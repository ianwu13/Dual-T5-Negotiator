import numpy as np
import matplotlib.pyplot as plt

lr_dir = './'

f = open(f'{lr_dir}plot_data.txt', 'r')
epochs_eval = []
eval_loss = []
bleu1 = []
bleu2 = []
bs_p = []
bs_r = []
bs_f1 = []
line = f.readline()
while line != '':
    tmp = line.split(',')
    eval_loss.append(float(tmp[0].split(':')[1]))
    bleu1.append(float(tmp[1].split(':')[1]))
    bleu2.append(float(tmp[2].split(':')[1]))
    bs_p.append(float(tmp[3].split(':')[1]))
    bs_r.append(float(tmp[4].split(':')[1]))
    bs_f1.append(float(tmp[5].split(':')[1]))
    epochs_eval.append(float(tmp[-1].split(':')[1].replace('}', '')))
    line = f.readline()


f = open(f'{lr_dir}plot_train_loss_data.txt', 'r')
epochs_train = []
train_loss = []
line = f.readline()
while line != '':
    train_loss.append(float(line.split(',')[0].split(':')[1]))
    epochs_train.append(float(line.replace('}', '').split(':')[-1]))
    line = f.readline()


# Plot BLEU seperately for good scale
plt.plot(epochs_eval, eval_loss, label='Eval Loss')
plt.plot(epochs_train, train_loss, label='Train Loss')

plt.title('Loss Over Time', y=1)
plt.ylabel('Loss')
plt.xlabel('Epochs Trained')
plt.legend()
plt.savefig(f'{lr_dir}eval_loss.png')
plt.clf()

plt.plot(epochs_eval, bs_p, label='BERTScore Precision')
plt.plot(epochs_eval, bs_r, label='BERTScore Recall')
plt.plot(epochs_eval, bs_f1, label='BERTScore F1')

plt.title('BERTScores Over Time', y=1)
plt.ylabel('Score')
plt.xlabel('Epochs Trained')
plt.legend()
plt.savefig(f'{lr_dir}bertscores.png')
plt.clf()

# Plot BLEU seperately for good scale
plt.plot(epochs_eval, bleu1, label='Bleu-1 Score')
plt.plot(epochs_eval, bleu2, label='Bleu-2 Score')

plt.title('Bleu Scores Over Time', y=1)
plt.ylabel('Score')
plt.xlabel('Epochs Trained')
plt.legend()
plt.savefig(f'{lr_dir}bleuscore.png')
