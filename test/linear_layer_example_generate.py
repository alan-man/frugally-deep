#!/usr/bin/env python3

import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


#OR
inputs = Input(shape=(2,))
x = Dense(5, activation='relu')(inputs)
predictions = Dense(1, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='nadam')

model.fit(
    np.asarray([[1,1], [0,0], [1,0], [0,1]]),
    np.asarray([[1], [0], [1], [1]]), epochs=10)

model.save('example_linear_model.keras')


#AND
inputs = Input(shape=(2,))
x = Dense(5, activation='relu')(inputs)
predictions = Dense(1, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='nadam')

model.fit(
    np.asarray([[1,1], [0,0], [1,0], [0,1]]),
    np.asarray([[1], [0], [0], [0]]), epochs=10)

model.save('example_linear_model.keras')


#XOR
inputs = Input(shape=(2,))
x = Dense(5, activation='relu')(inputs)
predictions = Dense(1, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='nadam')

model.fit(
    np.asarray([[1,1], [0,0], [1,0], [0,1]]),
    np.asarray([[0], [0], [1], [1]]), epochs=10)

model.save('example_linear_model.keras')




