import numpy as np 
import tensorflow as tf 

def cnn(params, nb_inputs, nb_outputs):
    """
    Routine serve to build a convolutional neural network
    """

    inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

    conv = inputs

    if params.iflo_activation == "LeakyReLU":
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    else:
        activation = getattr(tf.keras.layers,params.iflo_activation)()      

    for i in range(int(params.iflo_nb_layers)):
        conv = tf.keras.layers.Conv2D(
            filters=params.iflo_nb_out_filter,
            kernel_size=(params.iflo_conv_ker_size, params.iflo_conv_ker_size),
            kernel_initializer=params.iflo_weight_initialization,
            padding="same",
        )(conv)

        conv = activation(conv)

        conv = tf.keras.layers.Dropout(params.iflo_dropout_rate)(conv)

    outputs = conv

    outputs = tf.keras.layers.Conv2D(
        filters=nb_outputs,
        kernel_size=(
            1,
            1,
        ),
        kernel_initializer=params.iflo_weight_initialization,
        activation=None,
    )(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def unet(params, nb_inputs, nb_outputs):
    """
    Routine serve to define a UNET network from keras_unet_collection
    """

    from keras_unet_collection import models

    layers = np.arange(int(params.iflo_nb_blocks))

    number_of_filters = [
        params.iflo_nb_out_filter * 2 ** (layers[i]) for i in range(len(layers))
    ]

    return models.unet_2d(
        (None, None, nb_inputs),
        number_of_filters,
        n_labels=nb_outputs,
        stack_num_down=2,
        stack_num_up=2,
        activation=params.iflo_activation,
        output_activation=None,
        batch_norm=False,
        pool="max",
        unpool=False,
        name="unet",
    )



def fnn(params):
    """
    Build a feed-forward neural network for the inversion
    """

    nb_layers = 5 #params.opti_fnn_layers
    nb_neurons = 30 #params.opti_fnn_neurons
    activation = "leaky_relu"

    nb_inputs = len(params.opti_fnn_inputs)
    # nb_outputs = 22 # for the moment, actually > len(params.opti_control) + nz*2 oder so
    nb_outputs = len(params.opti_fnn_outputs)
    if "vel" in params.opti_fnn_outputs:
        nb_outputs -= 1
        nb_outputs += 2 * params.iflo_Nz
    # nb_inputs > 2 for x,y
    # nb_outputs > 22 for usurf, slidingco, u(z1), u(z2),..., v(z1), v(z2),...

    inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs]) # state.dY and state.dX stacked on top of each other shape(y,x,2)
    nn = inputs

    for i in range(int(nb_layers)):
        nn = tf.keras.layers.Dense(nb_neurons, activation=activation)(nn)

    outputs = tf.keras.layers.Dense(nb_outputs, activation=activation)(nn)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)