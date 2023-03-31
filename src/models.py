import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mlp_builder(hp_n_hidden_layers: int, hp_units_mult: int, hp_batch_normalization: bool,
                hp_input_batch_normalization: bool, hp_dropout: float) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(24))
    if hp_input_batch_normalization:
        model.add(layers.BatchNormalization())

    n_units = 3 * (2 ** (hp_n_hidden_layers - 1)) * hp_units_mult
    for _ in range(hp_n_hidden_layers):
        model.add(layers.Dense(n_units, activation='relu'))
        n_units //= 2
        if hp_batch_normalization:
            model.add(layers.BatchNormalization())
        if hp_dropout > 0:
            model.add(layers.Dropout(hp_dropout))

    model.add(layers.Dense(1))
    return model


def hp_mlp_builder(hp: kt.HyperParameters) -> keras.Model:
    hp_n_hidden_layers = hp.Int("n_hidden_layers", min_value=1, max_value=6, step=1, default=2)
    hp_units_mult = hp.Choice("units_mult", values=[1, 2, 4, 8, 16, 32], default=4)
    hp_batch_normalization = hp.Boolean("batch_normalization", default=False)
    hp_input_batch_normalization = hp.Boolean("input_batch_normalization", default=False)
    hp_dropout = hp.Choice("dropout", values=[0.0, 0.2, 0.5])

    return mlp_builder(hp_n_hidden_layers, hp_units_mult, hp_batch_normalization, hp_input_batch_normalization,
                       hp_dropout)


def optimal_mlp_builder() -> keras.Model:
    pass


def convnet_builder(hp_n_conv_blocks: int, hp_n_conv_layers: int, hp_filters_mult: int, hp_conv_spatial_dropout: float,
                    hp_mlp_n_hidden_layers: int, hp_mlp_units_mult: int, hp_mlp_dropout: float,
                    hp_batch_normalization: bool, hp_input_batch_normalization: bool) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(24))
    if hp_input_batch_normalization:
        model.add(layers.BatchNormalization())

    model.add(layers.Reshape((-1, 1)))

    # Convolutional network
    n_filters = 8 * hp_filters_mult
    for _ in range(hp_n_conv_blocks):  # block
        for _ in range(hp_n_conv_layers):  # layer
            model.add(layers.Conv1D(n_filters, 3, padding='same', activation='relu'))
            if hp_batch_normalization:
                model.add(layers.BatchNormalization())
            if hp_conv_spatial_dropout:
                model.add(layers.SpatialDropout1D(hp_conv_spatial_dropout))

        model.add(layers.MaxPooling1D())
        n_filters *= 2

    model.add(layers.Flatten())

    # MLP at the end
    if hp_mlp_n_hidden_layers > 0:
        n_units = 4 * (2 ** (hp_mlp_n_hidden_layers - 1)) * hp_mlp_units_mult
        for _ in range(hp_mlp_n_hidden_layers):
            model.add(layers.Dense(n_units, activation='relu'))
            n_units //= 2
            if hp_batch_normalization:
                model.add(layers.BatchNormalization())
            if hp_mlp_dropout > 0:
                model.add(layers.Dropout(hp_mlp_dropout))

    model.add(layers.Dense(1))

    return model


def hp_convnet_builder(hp: kt.HyperParameters) -> keras.Model:
    # Convolutional network params
    hp_n_conv_blocks = hp.Int("n_conv_blocks", min_value=1, max_value=3, step=1)
    hp_n_conv_layers = hp.Int("n_conv_layers", min_value=1, max_value=3, step=1)
    hp_filters_mult = hp.Choice("conv_filters_mult", values=[1, 2, 4, 8])
    hp_conv_spatial_dropout = hp.Choice("conv_spatial_dropout", values=[0.0, 0.1, 0.2])

    # MLP at the end params
    hp_mlp_n_hidden_layers = hp.Int("n_mlp_hidden_layers", min_value=0, max_value=3, step=1, default=0)
    hp_mlp_units_mult, hp_mlp_dropout = None, None
    if hp_mlp_n_hidden_layers > 0:
        hp_mlp_units_mult = hp.Choice("mlp_units_mult", values=[1, 2, 4, 8, 16], default=4)
        hp_mlp_dropout = hp.Choice("mlp_dropout", values=[0.0, 0.2, 0.5])

    # Other params
    hp_batch_normalization = hp.Boolean("batch_normalization", default=False)
    hp_input_batch_normalization = hp.Boolean("input_batch_normalization", default=False)

    return convnet_builder(hp_n_conv_blocks, hp_n_conv_layers, hp_filters_mult, hp_conv_spatial_dropout,
                           hp_mlp_n_hidden_layers, hp_mlp_units_mult, hp_mlp_dropout, hp_batch_normalization,
                           hp_input_batch_normalization)


def optimal_convnet_builder() -> keras.Model:
    pass


def _conv_block(x, n_filters, kernel_size: int = 2, n_conv_layers: int = 1, batch_normalization: bool = False,
                spatial_dropout: float = 0):
    for _ in range(n_conv_layers):
        x = layers.Conv1D(n_filters, kernel_size, activation='relu', padding='same')(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if spatial_dropout > 0:
            x = layers.SpatialDropout1D(spatial_dropout)(x)
    skip = x
    x = layers.MaxPooling1D()(x)
    return skip, x


def _deconv_block(x, skip, n_filters, kernel_size: int = 3, n_conv_layers: int = 1, batch_normalization: bool = False,
                  spatial_dropout: float = 0):
    x = layers.UpSampling1D()(x)
    x = layers.Conv1D(n_filters, 1, activation='linear')(x)
    x = layers.Concatenate()([skip, x])
    for _ in range(n_conv_layers):
        x = layers.Conv1D(n_filters, kernel_size, activation='relu', padding='same')(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if spatial_dropout > 0:
            x = layers.SpatialDropout1D(spatial_dropout)(x)
    return x


def unet_builder(hp_unet_depth: int, hp_n_conv_layers: int, hp_filters_mult: int, hp_spatial_dropout: float,
                 hp_batch_normalization: bool, hp_input_batch_normalization: bool) -> keras.Model:
    inputs = layers.Input(24)
    x = layers.Reshape((-1, 1))(inputs)
    if hp_input_batch_normalization:
        x = layers.BatchNormalization()(x)

    n_filters = 8 * hp_filters_mult
    skip_layers = []

    # Encoder
    for _ in range(hp_unet_depth):
        skip, x = _conv_block(x, n_filters, n_conv_layers=hp_n_conv_layers, batch_normalization=hp_batch_normalization,
                              spatial_dropout=hp_spatial_dropout)
        n_filters *= 2
        skip_layers.append(skip)

    # Bottleneck
    x, _ = _conv_block(x, n_filters)

    # Decoder
    for _ in range(hp_unet_depth):
        n_filters //= 2
        x = _deconv_block(x, skip_layers.pop(), n_filters, n_conv_layers=hp_n_conv_layers,
                          batch_normalization=hp_batch_normalization, spatial_dropout=hp_spatial_dropout)

    x = layers.Conv1D(1, 1, activation='linear')(x)

    outputs = layers.Flatten()(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def hp_unet_builder(hp: kt.HyperParameters) -> keras.Model:
    hp_unet_depth = hp.Int("unet_depth", min_value=0, max_value=3, step=1, default=2)
    hp_n_conv_layers = hp.Int("n_conv_layers", min_value=1, max_value=3, step=1)
    hp_filters_mult = hp.Choice("conv_filters_mult", values=[1, 2, 4, 8, 16], default=4)
    hp_spatial_dropout = hp.Choice("conv_spatial_dropout", values=[0.0, 0.1, 0.2])
    hp_batch_normalization = hp.Boolean("batch_normalization", default=False)
    hp_input_batch_normalization = hp.Boolean("input_batch_normalization", default=False)

    return unet_builder(hp_unet_depth, hp_n_conv_layers, hp_filters_mult, hp_spatial_dropout, hp_batch_normalization,
                        hp_input_batch_normalization)


def optimal_unet_builder() -> keras.Model:
    pass


def optimal_model_builder() -> keras.Model:
    pass
