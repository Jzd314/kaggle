import pandas as pd
import numpy as np
import os

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

test_df.head()
test_df.info()
train_df.head()
train_df.info()
test_df.isnull().sum()
train_df.isnull().sum()

test_df.drop(["keyword","location"],axis = 1, inplace = True)
train_df.drop(["keyword","location"],axis = 1, inplace = True)

import text_hammer as th
from tqdm._tqdm_notebook import tqdm_notebook 


def text_preprocessing(df,col_name):
    column = col_name
    df[column] = df[column].progress_apply(lambda x :str(x))
    df[column] = df[column].progress_apply(lambda x :th.remove_emails(x))
    df[column] = df[column].progress_apply(lambda x :th.remove_html_tags(x))
    df[column] = df[column].progress_apply(lambda x :th.remove_special_chars(x))
    df[column] = df[column].progress_apply(lambda x :th.remove_accented_chars(x))
    
    return(df)

train_cleaned_df = text_preprocessing(train_df,'text')
train_cleaned_df[train_cleaned_df.target == 0]

from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
bert = TFBertModel.from_pretrained('bert-large-uncased')

print("max len of tweets",max([len(x.split()) for x in train_df.text]))


x_train = tokenizer(
    text = train_cleaned_df.text.tolist(),
    add_special_tokens = True,
    max_length = 35,
    truncation = True, #超過35單字刪除
    padding = True,  #不足35單字補0
    return_tensors = "tf",
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

y_train = train_cleaned_df.target.values

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy,BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy,BinaryAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

max_len = 35
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

embeddings = bert([input_ids,input_mask])[1]

out = tf.keras.layers.Dropout(0.1)(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)

y = Dense(1,activation = 'sigmoid')(out)

model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True

model.summary()

optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=6e-06, 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

loss = BinaryCrossentropy(from_logits = True)
metric = BinaryAccuracy('accuracy'),

model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

train_history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = y_train,
    validation_split = 0.2,
    epochs=1, 
    batch_size=24
)

test_cleaned_df = text_preprocessing(test_df,'text')

x_test = tokenizer(
    text=test_cleaned_df.text.tolist(),
    add_special_tokens=True,
    max_length=35,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

predicted = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})

y_predicted = np.where(predicted>0.5,1,0)

sample_df = pd.read_csv("sample_submission.csv")

sample_df['id'] = test_df.id
sample_df['target'] = y_predicted

sample_df.to_csv('1.csv',index = False)