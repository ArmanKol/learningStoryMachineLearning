from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('seconds')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

learningData = pd.read_csv("test.csv")

learningData['datum'] = learningData['datum'].astype('datetime64')
learningData['tijd'] = learningData['datum'].dt.time
learningData['datum'] = learningData['datum'].dt.date
learningData['tijd'] = learningData['tijd'].astype('str')

seconds = []

for i in learningData['tijd']:
    seconds.append(get_sec(i))

for j in learningData['stand']:
    if ('ON' in j):
        learningData['stand'] = learningData['stand'].str.replace('ON', '1')
    if ('OFF' in j):
        learningData['stand'] = learningData['stand'].str.replace('OFF', '0')

for k in learningData['datum']:
    learningData['dag'] = k.weekday()+1

learningData['stand'] = learningData['stand'].astype('int64')
learningData['seconds'] = seconds
learningData = learningData.drop(columns="datum")
learningData = learningData.drop(columns="tijd")

train, test = train_test_split(learningData, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#Input pipeline
batch_size = 32 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

print(train_ds)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  #print('A batch of ages:', feature_batch['seconds'])
  print('A batch of targets:', label_batch )


feature_columns = []

# numeric cols
for header in ['kamer', 'stand', 'dag']:
  feature_columns.append(feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)
