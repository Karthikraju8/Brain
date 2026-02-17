"""
X-Farmer Model Architecture for Brain Tumor Classification

A custom CNN architecture optimized for medical image analysis
with enhanced feature extraction capabilities.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.regularizers import l2
from typing import Tuple, List, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    IMAGE_SIZE, CHANNELS, NUM_CLASSES,
    XFARMER_CONFIG, LEARNING_RATE, OPTIMIZER
)


class ConvBlock(layers.Layer):
    """
    Convolutional block with Conv2D, BatchNorm, and Activation
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        use_batch_norm: bool = True,
        activation: str = "relu",
        kernel_regularizer: Optional[float] = None,
        name: str = None,
        **kwargs
    ):
        super(ConvBlock, self).__init__(name=name, **kwargs)

        regularizer = l2(kernel_regularizer) if kernel_regularizer else None

        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_regularizer=regularizer,
            use_bias=not use_batch_norm
        )

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = layers.BatchNormalization()

        self.activation = layers.Activation(activation)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        if self.use_batch_norm:
            x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class ResidualBlock(layers.Layer):
    """
    Residual block with skip connection for better gradient flow
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int] = (3, 3),
        use_batch_norm: bool = True,
        name: str = None,
        **kwargs
    ):
        super(ResidualBlock, self).__init__(name=name, **kwargs)

        self.conv1 = ConvBlock(
            filters, kernel_size, use_batch_norm=use_batch_norm
        )
        self.conv2 = layers.Conv2D(
            filters, kernel_size, padding="same", use_bias=not use_batch_norm
        )
        self.bn = layers.BatchNormalization() if use_batch_norm else None
        self.use_batch_norm = use_batch_norm

        # Skip connection projection (if dimensions change)
        self.skip_conv = None
        self.skip_bn = None

    def build(self, input_shape):
        if input_shape[-1] != self.conv2.filters:
            self.skip_conv = layers.Conv2D(
                self.conv2.filters, (1, 1), padding="same"
            )
            if self.use_batch_norm:
                self.skip_bn = layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x)
        if self.bn:
            x = self.bn(x, training=training)

        # Skip connection
        skip = inputs
        if self.skip_conv:
            skip = self.skip_conv(inputs)
            if self.skip_bn:
                skip = self.skip_bn(skip, training=training)

        x = layers.add([x, skip])
        x = layers.Activation("relu")(x)
        return x


class AttentionBlock(layers.Layer):
    """
    Spatial attention mechanism for focusing on tumor regions
    """

    def __init__(self, filters: int, name: str = None, **kwargs):
        super(AttentionBlock, self).__init__(name=name, **kwargs)

        self.conv1 = layers.Conv2D(filters // 4, (1, 1), activation="relu")
        self.conv2 = layers.Conv2D(filters // 4, (3, 3), padding="same", activation="relu")
        self.conv3 = layers.Conv2D(1, (1, 1), activation="sigmoid")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        attention_map = self.conv3(x)
        return inputs * attention_map


class XFarmerModel:
    """
    X-Farmer: eXtended Feature Aggregation and Recognition Model for MRI

    A custom CNN architecture designed for brain tumor classification
    with the following features:
    - Multi-scale feature extraction
    - Residual connections
    - Spatial attention mechanisms
    - Batch normalization for stable training
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, CHANNELS),
        num_classes: int = NUM_CLASSES,
        config: dict = None
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config or XFARMER_CONFIG
        self.model = None

    def build(self) -> Model:
        """Build the X-Farmer model architecture"""

        inputs = layers.Input(shape=self.input_shape, name="input_image")

        # Initial convolution
        x = ConvBlock(
            self.config["conv_filters"][0],
            self.config["kernel_size"],
            use_batch_norm=self.config["use_batch_norm"],
            name="initial_conv"
        )(inputs)

        # Feature extraction blocks with pooling
        for i, filters in enumerate(self.config["conv_filters"][1:], 1):
            # Residual block
            x = ResidualBlock(
                filters,
                self.config["kernel_size"],
                use_batch_norm=self.config["use_batch_norm"],
                name=f"residual_block_{i}"
            )(x)

            # Additional conv block
            x = ConvBlock(
                filters,
                self.config["kernel_size"],
                use_batch_norm=self.config["use_batch_norm"],
                name=f"conv_block_{i}"
            )(x)

            # Attention mechanism (on deeper layers)
            if i >= 2:
                x = AttentionBlock(filters, name=f"attention_{i}")(x)

            # Max pooling
            x = layers.MaxPooling2D(
                pool_size=self.config["pool_size"],
                name=f"pool_{i}"
            )(x)

        # Global feature aggregation
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

        # Dense layers
        for i, units in enumerate(self.config["dense_units"]):
            x = layers.Dense(units, name=f"dense_{i}")(x)
            x = layers.BatchNormalization(name=f"bn_dense_{i}")(x)
            x = layers.Activation("relu", name=f"relu_dense_{i}")(x)
            x = layers.Dropout(
                self.config["dropout_rate"],
                name=f"dropout_{i}"
            )(x)

        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
            name="output"
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs, name="XFarmer")

        return self.model

    def compile(
        self,
        learning_rate: float = LEARNING_RATE,
        optimizer: str = OPTIMIZER
    ):
        """Compile the model with optimizer and loss"""

        if self.model is None:
            self.build()

        # Select optimizer
        if optimizer.lower() == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == "sgd":
            opt = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif optimizer.lower() == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )

        return self.model

    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build()
        self.model.summary()

    def get_model(self) -> Model:
        """Get the compiled model"""
        if self.model is None:
            self.build()
            self.compile()
        return self.model


def build_xfarmer_model(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, CHANNELS),
    num_classes: int = NUM_CLASSES,
    config: dict = None,
    compile_model: bool = True
) -> Model:
    """
    Convenience function to build X-Farmer model

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        config: Model configuration
        compile_model: Whether to compile the model

    Returns:
        Compiled Keras Model
    """
    xfarmer = XFarmerModel(input_shape, num_classes, config)
    model = xfarmer.build()

    if compile_model:
        xfarmer.compile()

    return model


def build_xfarmer_lite(
    input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, CHANNELS),
    num_classes: int = NUM_CLASSES
) -> Model:
    """
    Build a lighter version of X-Farmer for faster training

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes

    Returns:
        Compiled Keras Model
    """
    lite_config = {
        "conv_filters": [16, 32, 64, 128],
        "kernel_size": (3, 3),
        "pool_size": (2, 2),
        "dense_units": [256, 128],
        "dropout_rate": 0.4,
        "use_batch_norm": True,
    }

    return build_xfarmer_model(input_shape, num_classes, lite_config)


if __name__ == "__main__":
    # Test model building
    print("Building X-Farmer Model...")
    print("=" * 50)

    model = build_xfarmer_model()
    model.summary()

    print("\n" + "=" * 50)
    print("Building X-Farmer Lite Model...")
    print("=" * 50)

    lite_model = build_xfarmer_lite()
    lite_model.summary()

    # Test forward pass
    import numpy as np
    dummy_input = np.random.rand(1, *IMAGE_SIZE, CHANNELS).astype(np.float32)
    output = model.predict(dummy_input)
    print(f"\nTest prediction shape: {output.shape}")
    print(f"Test prediction: {output}")
