# -*- coding: utf-8 -*-

#导入相应的类
import re
import numpy as np
import pandas as pd
import random

from collections import defaultdict

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

from sklearn.preprocessing import StandardScaler
from string import punctuation
from functools import partial
import tensorflow as tf

np.random.seed(110)
random.seed(110)


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)
tf.app.flags.DEFINE_integer('lstm_units', 193, '')
tf.app.flags.DEFINE_integer('dense_units', 136, '')
tf.app.flags.DEFINE_float('lstm_dropout', 0.19, '')
tf.app.flags.DEFINE_float('dense_dropout', 0.18, '')
tf.app.flags.DEFINE_integer('leaks_dense_units', 68, '')
tf.app.flags.DEFINE_string('optimizer', "adam", '')
tf.app.flags.DEFINE_float('lr', 0.001, '')
tf.app.flags.DEFINE_string('d_train_dir', ".", 'Train-data dir')
tf.app.flags.DEFINE_string('d_result_dir', ".", 'Result-data dir')
FLAGS = tf.app.flags.FLAGS

lstm_units = FLAGS.lstm_units
dense_units = FLAGS.dense_units
lstm_dropout = FLAGS.lstm_dropout
dense_dropout = FLAGS.dense_dropout
leaks_dense_units = FLAGS.leaks_dense_units
optimizer = FLAGS.optimizer
lr = FLAGS.lr
d_train_dir = FLAGS.d_train_dir
d_result_dir = FLAGS.d_result_dir

EMBEDDING_FILE = d_train_dir + '/glove.840B.300d.txt'
TRAIN_DATA_FILE = d_train_dir + '/train.csv'
TEST_DATA_FILE = d_train_dir + '/test.csv'
MAX_SEQ_LENGTH = 30
MAX_NUMBER_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

print('start ')
print("lstm_units:", lstm_units)
print("dense_units:", dense_units)
print("lstm_dropout:", lstm_dropout)
print("dense_dropout:", dense_dropout)
print("leaks_dense_units:", leaks_dense_units)
print("optimizer:", optimizer)
print("lr:", lr)

#初始化语料库
embeddings_index = {}
with open(EMBEDDING_FILE, 'r', encoding="utf-8") as f:
    count = 0
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError as ve:
            print('ignore error ' + str(ve))

print('word vectors in glove: %d' % len(embeddings_index))

#定义数据清洗方法
stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']


# 数据清洗参考https://www.kaggle.com/currie32/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stop_words=True):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower()\
                .replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
                .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
                .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
                .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
                .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
                .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
                .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")

    text = text.split()

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', text)
    text = re.sub(r"([0-9]+)000000", r"\1m", text)
    text = re.sub(r"([0-9]+)000", r"\1k", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)


    # Return a list of words
    return text

parse_apply_func = partial(text_to_wordlist, remove_stop_words=False)

#数据清洗
train_df = pd.read_csv(TRAIN_DATA_FILE)
train_df["question1"] = train_df["question1"].fillna("").apply(parse_apply_func)
train_df["question2"] = train_df["question2"].fillna("").apply(parse_apply_func)
texts_1 = train_df["question1"].tolist()
texts_2 = train_df["question2"].tolist()
labels = train_df["is_duplicate"].tolist()
print('train.csv length:%d' % len(texts_1))

test_df = pd.read_csv(TEST_DATA_FILE)
test_df["question1"] = test_df["question1"].fillna("").apply(parse_apply_func)
test_df["question2"] = test_df["question2"].fillna("").apply(parse_apply_func)
test_texts_1 = test_df["question1"].tolist()
test_texts_2 = test_df["question2"].tolist()
test_ids = test_df["test_id"].tolist()
print('test.csv length:%d' % len(test_texts_1))

# 生成主要特征
tokenizer = Tokenizer(num_words=MAX_NUMBER_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('tokens:', str(len(word_index)))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQ_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQ_LENGTH)
labels = np.array(labels)
print('data_1.shape:', data_1.shape)
print('labels.shape:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQ_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQ_LENGTH)
test_ids = np.array(test_ids)

# 生成弱特征
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

mix_df = pd.concat([train_df[['question1', 'question2']],
                    test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(mix_df.shape[0]):
    q_dict[mix_df.question1[i]].add(mix_df.question2[i])
    q_dict[mix_df.question2[i]].add(mix_df.question1[i])


def q1_freq(row):
    return len(q_dict[row['question1']])


def q2_freq(row):
    return len(q_dict[row['question2']])


def q1_q2_intersect(row):
    return len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))


train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)
train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)

test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_freq'] = test_df.apply(q1_freq, axis=1, raw=True)
test_df['q2_freq'] = test_df.apply(q2_freq, axis=1, raw=True)

leaks = train_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
test_leaks = test_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]

ss = StandardScaler()
ss.fit(np.vstack((leaks, test_leaks)))
leaks = ss.transform(leaks)
test_leaks = ss.transform(test_leaks)

## 生成嵌入层
nb_words = min(MAX_NUMBER_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

# 切分数据
index_array = [*range(len(data_1))]
random.shuffle(index_array)
idx_train = index_array[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
idx_val = index_array[int(len(data_1) * (1 - VALIDATION_SPLIT)):]

# 这里需要将data_1和data_2，双向stack一下，为了解决数据对称性问题
data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
leaks_train = np.vstack((leaks[idx_train], leaks[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
leaks_val = np.vstack((leaks[idx_val], leaks[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

'''
标签重新分配权重
参考
https://github.com/howardyclo/Kaggle-Quora-Question-Pairs#class-label-reweighting
'''
validation_weight = np.ones(len(labels_val))
validation_weight *= 0.471544715
validation_weight[labels_val == 0] = 1.30903328
model_class_weight = {0: 1.30903328, 1: 0.471544715}

# 构建神经网络
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTH,
                            trainable=False)
lstm_layer = LSTM(lstm_units, dropout=lstm_dropout, recurrent_dropout=lstm_dropout)

sequence_input1 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_input1)
x1 = lstm_layer(embedded_sequences_1)

sequence_input2 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_input2)
y1 = lstm_layer(embedded_sequences_2)

leaks_input = Input(shape=(leaks.shape[1],))
leaks_dense = Dense(leaks_dense_units, activation='relu')(leaks_input)

merged = concatenate([x1, y1, leaks_dense])
merged = BatchNormalization()(merged)
merged = Dropout(dense_dropout)(merged)

merged = Dense(dense_units, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dropout(dense_dropout)(merged)

preds = Dense(1, activation='sigmoid')(merged)

my_optimizer = optimizers.Adam(lr=lr)
if optimizer == 'sgd':
    my_optimizer = optimizers.SGD(lr=lr)
elif optimizer == 'nadam':
    my_optimizer = optimizers.Nadam(lr=lr)

model = Model(inputs=[sequence_input1, sequence_input2, leaks_input], outputs=preds)
model.compile(loss='binary_crossentropy', optimizer=my_optimizer, metrics=['accuracy'])
model.summary()

#开始训练
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
best_model_path = d_result_dir + '/best_model.h5'
model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
hist = model.fit([data_1_train, data_2_train, leaks_train], labels_train,
                 validation_data=([data_1_val, data_2_val, leaks_val], labels_val, validation_weight),
                 epochs=200, batch_size=4096, shuffle=True,
                 callbacks=[early_stopping, model_checkpoint],
                 class_weight=model_class_weight, verbose=2)

model.load_weights(best_model_path)
best_val_score = min(hist.history['val_loss'])
print("best_val_score:%f" % best_val_score)

# 由于输入层含有双向特征，这里需要Fine tune一下
preds = model.predict([test_data_1, test_data_2, test_leaks], batch_size=8192)
preds += model.predict([test_data_2, test_data_1, test_leaks], batch_size=8192)
preds /= 2

# 保存结果
submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
submission.to_csv(d_result_dir + '/submission.csv', index=False)
