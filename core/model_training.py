import tensorflow as tf


def load_images(directory_path="data/Data for test", image_size=(128, 128)):
    train_data, val_data = tf.keras.utils.image_dataset_from_directory(
        directory_path,
        validation_split=0.2,
        subset="both",
        seed=2023,
        batch_size=32,
        image_size=image_size,
        label_mode="categorical"
    )
    return train_data, val_data


def create_data_augmentation_pipeline():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1)
    ])
    return data_augmentation


def create_image_classifier(image_size):
    classifier = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=image_size + (3,)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='swish'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='swish'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='swish'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='swish'),
        tf.keras.layers.Dense(3)
    ])
    return classifier


def train_image_classifier(
        data_directory="data/Data for test",
        save_to_directory="trained_model/",
        image_size=(128, 128),
        n_epochs=20
    ):

    train_data, val_data = load_images(
        directory_path=data_directory, image_size=image_size
    )

    data_augmentation = create_data_augmentation_pipeline()
    train_data = train_data.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    )

    image_classifier = create_image_classifier(image_size)
    image_classifier.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = image_classifier.fit(
        train_data,
        validation_data=val_data,
        epochs=n_epochs
    )

    image_classifier.save(save_to_directory + "image_classifier.h5")

    return image_classifier, history
