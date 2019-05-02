from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Каталог с данными для обучения, проверки, тестирования
train_dir = 'Training_4'
val_dir = 'Training_4'
test_dir = 'Testing_4'
# Размеры изображения
img_width, img_height = 250, 250
# Tensorflow, channels_last
input_shape = (img_width, img_height, 3)

epochs = 30  # Количество эпох
batch_size = 2  # Размер мини-выборки
nb_train_samples = 14  # Количество изображений для обучения
nb_validation_samples = 14  # Количество изображений для проверки
nb_test_samples = 2  # Количество изображений для тестирования


def create_nn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def create_generator(gen, dir_name):
    generator = gen.flow_from_directory(
        dir_name,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    return generator


def fit_nn(loc_model, train_generator, val_generator):
    loc_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size)
    return loc_model


def get_data_and_label_from_gen(gen):
    x, y = zip(*(gen[i] for i in range(len(gen))))
    x_value, y_value = np.vstack(x), np.vstack(y)
    return x_value, y_value.reshape(-1)


def get_real_label(nlabel):
    return "Ярослав" if nlabel == 1 else "Эйдан"


def show_graph(image, orig_label, predict_label):
    for i, img in enumerate(image):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img.reshape(img_width, img_height, 3), cmap='gray', interpolation='none')
        plt.title("Class: {} Predict: {}".format(orig_label[i], predict_label[i]))
        plt.xlabel(get_real_label(predict_label[i]))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Create data generator
    data_gen = ImageDataGenerator(rescale=1. / 255)
    train_gen = create_generator(data_gen, train_dir)
    val_gen = create_generator(data_gen, val_dir)
    test_gen = create_generator(data_gen, test_dir)

    # Load or create nn
    try:
        model = load_model("model.h5py")
    except (OSError, ImportError, ValueError):
        model = create_nn()
        model = fit_nn(model, train_gen, val_gen)
        model.save("model.h5py")

    # Predict
    test_x, test_y = get_data_and_label_from_gen(test_gen)
    predict = np.round(model.predict(test_x, batch_size=batch_size)).reshape(-1)
    print("Исходная разметка: {} \nПредсказананная: {}".format(test_y, predict))

    # Show results
    scores = model.evaluate_generator(test_gen)
    print("Точность: %.2f%%" % (scores[1] * 100))
    show_graph(test_x, test_y, predict)
