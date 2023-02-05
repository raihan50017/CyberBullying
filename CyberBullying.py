# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('New Data (CUET - Kaggle)/Bengali_dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



AUTO = tf.data.experimental.AUTOTUNE

# Data access
# GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')

# Configuration
EPOCHS = 3
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
MAX_LEN = 128


# First load the real tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')
tokenizer.pad_token = tokenizer.eos_token
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
# fast_tokenizer = SentencePieceBPETokenizer('vocab.txt')
# fast_tokenizer


def fast_encode_xlm(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.batch_encode_plus(text_chunk, pad_to_max_length = True, max_length = maxlen)
        all_ids.extend(np.array(encs.input_ids))
    
    return np.array(all_ids)


ben1=pd.read_csv("/kaggle/input/bengali-dataset/Bengali_dataset/iben/trac2_iben_dev.csv")
ben2=pd.read_csv("/kaggle/input/bengali-dataset/Bengali_dataset/iben/trac2_iben_train.csv")
test=pd.read_csv("/kaggle/input/bengali-dataset/Bengali_dataset/trac2-test-package/trac2_iben_test.csv")


ben1 = ben1.rename(columns={"ID": "id", "Text": "comment_text", 'Sub-task A': 'toxic'})
ben2 = ben2.rename(columns={"ID": "id", "Text": "comment_text", 'Sub-task A': 'toxic'})
test = test.rename(columns={"ID": "id", "Text": "comment_text"})
ben1.toxic = (ben1.toxic == 'OAG').astype(int)
ben2.toxic = (ben2.toxic == 'OAG').astype(int)


ben = pd.concat([
    ben1[['comment_text', 'toxic']],
    ben2[['comment_text', 'toxic']].query('toxic==1' or 'toxic==0')
])
ben.head()


ben["toxic"].value_counts()


from sklearn.model_selection import train_test_split
ben_train, ben_test= train_test_split(ben, test_size=0.2, random_state = 2020)

# hind_en = fast_encode_xlm(hind.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)
ben_train_en = fast_encode_xlm(ben_train.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)
ben_test_en = fast_encode_xlm(ben_test.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)
ben_testdata= fast_encode_xlm(test.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)


ben_test_en.shape


y_ben_train = ben_train.toxic.values
y_ben_test = ben_test.toxic.values


def build_model(transformer, max_len=128):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids, decoder_input_ids=input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    hidden_layer = Dense(32, activation = "relu") (cls_token)
    hidden_layer2 = Dense(16, activation = "relu") (hidden_layer)
    out = Dense(1, activation='sigmoid')(hidden_layer2)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


%%time
with strategy.scope():
    transformer_layer = (
        transformers.TFAutoModelForPreTraining.from_pretrained('t5-base')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()


for layer in model.layers:
    print("Layer:", layer, "\nTrainable = ", layer.trainable)


# model.layers[1].trainable = False


for layer in model.layers:
    print("Layer:", layer, "\nTrainable = ", layer.trainable)


ben_train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((ben_train_en, y_ben_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
ben_test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(ben_test_en)
    .batch(BATCH_SIZE)
)
# ben_test_dataset=(
#     tf.data.Dataset
#     .from_tensor_slices(ben_testdata)
#     .batch(BATCH_SIZE)
# )
ben_test_dataset_wlabel = (
    tf.data.Dataset
    .from_tensor_slices((ben_test_en, y_ben_test))
    .batch(BATCH_SIZE)
)


from sklearn.metrics import roc_auc_score
ben_test.toxic_predict = model.predict(ben_test_dataset, verbose=1)

roc_auc_score(y_true = ben_test.toxic, y_score = ben_test.toxic_predict)


from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, log_loss, f1_score
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 
threshold = Find_Optimal_Cutoff(ben_test.toxic, ben_test.toxic_predict)
# print("the optimal threshold is " + str(threshold[0]))
ben_test.toxic_predict_binary = [1 if p > threshold[0] else 0 for p in ben_test.toxic_predict]
th=threshold[0]


def plot_matrix(target, predicted_binary, name):
    print(target, predicted_binary)
    matrix = confusion_matrix(target, predicted_binary)
    TN, FP, FN, TP = matrix.ravel()
    print(TN, FP, FN, TP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if (TP + FP > 0) and (TP + FN > 0):
        F =  2 * (precision*recall) / (precision + recall)
    else:
        F = 0
    print(accuracy, precision, recall, F)
    cm_df = pd.DataFrame(matrix,
                         index = ['Nagative', 'Positive'], 
                         columns = ['Nagative', 'Positive'])
    subtitle = 'Precision ' + str(round(precision, 2)) + ' Recall ' + str(round(recall, 2))
    fig, ax = plt.subplots(figsize=(8,6))
    ax = sns.heatmap(cm_df, annot=True, fmt="d")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix - ' + name + "\n" + subtitle)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
plot_matrix(ben_test.toxic, ben_test.toxic_predict_binary, name = 'Bangla Offensive Language Detection')


model_10k = model
model_10k.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
n_steps = ben_train.shape[0]
train_history_10 = model_10k.fit(
    ben_train_dataset.repeat(),
    steps_per_epoch=n_steps,
    validation_data=ben_test_dataset_wlabel,
    epochs=5
)
ben_test10k = ben_test
ben_test10k.toxic_predict = model_10k.predict(ben_test_dataset, verbose=1)
# olid_test10k.to_csv('olid_test_10k.csv', index=False)

print('updated roc is ' + str(roc_auc_score(y_true = ben_test10k.toxic, y_score = ben_test10k.toxic_predict)))

threshold = Find_Optimal_Cutoff(ben_test10k.toxic, ben_test10k.toxic_predict)
print("the optimal threshold is " + str(threshold[0]))
ben_test10k.toxic_predict_binary = [1 if p > threshold[0] else 0 for p in ben_test10k.toxic_predict]

print('Updated f1-score is ' + str(f1_score(y_true = ben_test10k.toxic, y_pred = ben_test10k.toxic_predict_binary)))
plot_matrix(ben_test10k.toxic, ben_test10k.toxic_predict_binary, name = 'Bangla Offensive Language Detection')

del model_10k


(72 + 114)/(72 + 114 + 112 + 64)


threshold[0]


print('Updated f1-score is ' + str(f1_score(y_true = ben_test10k.toxic, y_pred = ben_test10k.toxic_predict_binary)))


(102+155) / (102+155+71+34)


# test['toxic'] = model.predict(ben_test_dataset, verbose=1)
# # testdata['toxic']=[1 if testdata['toxic'] > th else 0]                                                                                                         
# test['toxic']= [1 if p > th else 0 for p in test.toxic]
# test.to_csv('submission.csv', index=False)