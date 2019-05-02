from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import os
from skimage import color, io, filters


def load_sample(name_sample, type_sample, n_label):
    smpl, lbl = [], []
    os.chdir(type_sample + "/" + str(name_sample))
    img_list = os.listdir()
    for img in img_list:
        image = np.array(io.imread(img))
        smpl.append(image)
        lbl.append(n_label)
    os.chdir(os.getcwd() + "/../../")
    return smpl, lbl


def load_dataset():
    train_X_1, train_Y_1 = load_sample("me", "Training", 1)
    train_X_2, train_Y_2 = load_sample("Terner", "Training", 0)
    test_X_1, test_Y_1 = load_sample("me", "Testing", 1)
    test_X_2, test_Y_2 = load_sample("Terner", "Testing", 0)

    train_X_1.extend(train_X_2)
    train_Y_1.extend(train_Y_2)
    test_X_1.extend(test_X_2)
    test_Y_1.extend(test_Y_2)

    return (np.array(train_X_1), np.array(train_Y_1)), (np.array(test_X_1), np.array(test_Y_1))


# **Load the Data**
(train_X, train_Y), (test_X, test_Y) = load_dataset()
print("Тип массива: {}, его форма: {}".format(type(train_X), train_X.shape))
print("Тип элемента массива: {}, его форма: {}".format(type(train_X[0]), train_X[0].shape))

# **Analyze the Data**
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)


# **Show original data**

plt.figure(figsize=[5, 5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

plt.show()
# **Data Preprocessing**
"""
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X.shape, test_X.shape


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# **Split**

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

train_X.shape, valid_X.shape, train_label.shape, valid_label.shape

# **The Network**


batch_size = 64
epochs = 20
num_classes = 10

# **Neural Network Architecture**

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2), padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dense(num_classes, activation='softmax'))

# **Compile**

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
fashion_model.summary()

# **Train**

fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                  validation_data=(valid_X, valid_label))
# **Evalution**

test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# **Save model**

fashion_model.save("fashion_model_dropout.h5py")

# **Show valution**

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# **Predict Labels**

predicted_classes = fashion_model.predict(test_X)

# **Reshape** (0100000000 -> 1)

predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

# **Show results**

correct = np.where(predicted_classes == test_Y)[0]
print("Found {} correct labels".format(len(correct)))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes != test_Y)[0]
print("Found {} incorrect labels".format(len(incorrect)))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()

# **Classification Report**

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))
"""