from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mlp_builder(hp_n_hidden_layers: int, hp_units_mult: int, hp_unit_decrease_factor: float,
                hp_batch_normalization: bool, hp_input_batch_normalization: bool, hp_dropout: float) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(24))
    if hp_input_batch_normalization:
        model.add(layers.BatchNormalization())

    n_units_list = [3 * hp_units_mult]
    for _ in range(hp_n_hidden_layers - 1):
        n_units_list.append(int(n_units_list[-1] * hp_unit_decrease_factor))

    for n_units in n_units_list:
        model.add(layers.Dense(n_units, activation='relu'))
        if hp_batch_normalization:
            model.add(layers.BatchNormalization())
        if hp_dropout > 0:
            model.add(layers.Dropout(hp_dropout))

    model.add(layers.Dense(1))
    return model


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


def rnn_builder(hp_rnn_type: str, hp_n_neurons: int, hp_n_hidden_layers: int,
                hp_input_batch_normalization: bool) -> keras.Model:
    if hp_rnn_type == "lstm":
        RnnLayerType = layers.LSTM
    elif hp_rnn_type == "gru":
        RnnLayerType = layers.GRU
    else:
        raise ValueError(f"Wrong RNN type: {hp_rnn_type}")

    model = keras.Sequential()
    model.add(layers.Input(24))
    model.add(layers.Reshape((-1, 1))),
    if hp_input_batch_normalization:
        model.add(layers.BatchNormalization())

    for _ in range(hp_n_hidden_layers):
        model.add(RnnLayerType(hp_n_neurons, return_sequences=True))

    model.add(RnnLayerType(hp_n_neurons))

    model.add(layers.Dense(1))
    return model


@dataclass
class OptimalModelBuilders:
    mlp: Callable[[], keras.Model]
    convnet: Callable[[], keras.Model]
    unet: Callable[[], keras.Model]
    rnn: Callable[[], keras.Model]


optimal_model_builders_ch_2_11 = OptimalModelBuilders(
    mlp=lambda: mlp_builder(hp_n_hidden_layers=7, hp_units_mult=2, hp_unit_decrease_factor=1.0,
                            hp_batch_normalization=False, hp_input_batch_normalization=True, hp_dropout=0.0),

    convnet=lambda: convnet_builder(hp_n_conv_blocks=1, hp_n_conv_layers=2, hp_filters_mult=4,
                                    hp_conv_spatial_dropout=0.1, hp_mlp_n_hidden_layers=2, hp_batch_normalization=True,
                                    hp_input_batch_normalization=True, hp_mlp_units_mult=4, hp_mlp_dropout=0.2),

    unet=lambda: unet_builder(hp_unet_depth=3, hp_n_conv_layers=2, hp_filters_mult=4, hp_spatial_dropout=0.0,
                              hp_batch_normalization=False, hp_input_batch_normalization=True),

    rnn=lambda: rnn_builder(hp_rnn_type='gru', hp_n_neurons=128, hp_n_hidden_layers=0,
                            hp_input_batch_normalization=True)
)
