from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers

model1 = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.01)),
    Activation('relu'),
    Dropout(0.4),
    Dense(16, kernel_regularizer=regularizers.l2(0.04)),
    Activation('relu'),

    Dense(6),
    Activation('relu'),

    Dense(1, activation='sigmoid')
])
