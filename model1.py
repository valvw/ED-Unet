import tensorflow as tf

def conv_block(inputs, filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'):
    x = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def upconv_block(inputs, filters, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal'):
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def unet(input_shape=(256, 256, 3), num_classes=1, filters=[64, 128, 256, 512, 1024], kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Downsampling path
    c = inputs
    down_layers = []
    for i in range(len(filters)):
        c = conv_block(c, filters[i], kernel_size, activation, padding, kernel_initializer)
        down_layers.append(c)
        c = tf.keras.layers.MaxPooling2D((2, 2))(c)

    # Bottleneck layer
    c = conv_block(c, filters[-1], kernel_size, activation, padding, kernel_initializer)

    # Upsampling path
    for i in reversed(range(len(filters))):
        up = upconv_block(c, filters[i], kernel_size=2, strides=2, padding=padding, kernel_initializer=kernel_initializer)
        c = tf.keras.layers.concatenate([up, down_layers[i]])
        c = conv_block(c, filters[i], kernel_size, activation, padding, kernel_initializer)

    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=activation, kernel_initializer=kernel_initializer)(c)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model