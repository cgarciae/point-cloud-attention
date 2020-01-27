import typer
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
import cytoolz as cz
from sklearn.preprocessing import MinMaxScaler
import time

from . import transformer

np.random.seed(42)


def main(
    data_dir: Path = Path("data"),
    params_path: Path = Path("training/params.yml"),
    train_version: str = "max",
    test_version: str = "max",
    model: str = "attention",
):
    model_dir = f"models/{train_version}_{test_version}/{int(time.time())}"

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    X_train = np.load(data_dir / f"X_train_{train_version}.npy")
    y_train = np.load(data_dir / f"y_train.npy")
    X_test = np.load(data_dir / f"X_test_{test_version}.npy")
    y_test = np.load(data_dir / f"y_test.npy")

    preprocessor = MinMaxScaler(feature_range=(-1, 1))

    X_train, y_train = preprocess(X_train, y_train, params, preprocessor, mode="train")
    X_test, y_test = preprocess(X_test, y_test, params, preprocessor, mode="test")

    if model == "attention":
        model = AttentionModel(params)
    elif model == "pooling":
        model = PoolingModel(params)
    elif model == "set":
        model = SetTransformerModel(params)
    else:
        raise ValueError(f"Unknown model type: {model}")

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(lr=params["lr"]),
    )

    model(X_train[:2])
    model.summary()

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        steps_per_epoch=params["steps_per_epoch"],
        validation_data=(X_test, y_test),
        validation_steps=params["validation_steps"],
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(
                tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    [500, 1200], [0.001, 0.0005, 0.0001]
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_dir}/saved_model", save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir=model_dir),
        ],
    )


def preprocess(X, y, params, preprocessor, mode):

    for i in range(len(X)):
        paddings = X[i, :, -1] < 0
        X[i, paddings] = [-27, -27, -255]

    shape = X.shape
    X = X.reshape((-1, shape[-1]))

    if mode == "train":
        X = preprocessor.fit_transform(X)
    else:
        X = preprocessor.transform(X)

    X = X.reshape(shape)

    X = X.astype(np.float32)

    # X = X[: params["batch_size"] * 10]
    # y = y[: params["batch_size"] * 10]

    return X, y


class AttentionModel(tf.keras.Model):
    def __init__(self, params):
        super().__init__()

        input_features = params["input_features"]
        n_units = params["n_units"]
        n_heads = params["n_heads"]
        n_labels = params["n_labels"]
        n_units_att = params["n_units_att"]

        self.embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_units),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation("elu"),
            ]
            * params["n_dense_layers"]
        )

        self.self_attention_modules = tf.keras.Sequential(
            [AttentionModule(n_units_att, n_heads)] * params["n_attention_layers"]
        )

        self.dense_output = tf.keras.layers.Dense(n_labels, activation="softmax")

    @tf.function
    def call(self, x):
        batch_size = tf.shape(x)[0]

        token = tf.constant([-1.0, 1.0, -1.0])[None, None, :]
        token = tf.tile(token, (batch_size, 1, 1))

        x = tf.concat([token, x], axis=1)

        x = self.embeddings(x)
        x = self.self_attention_modules(x)
        x = x[:, 0]
        x = self.dense_output(x)

        return x


class SetTransformerModel(tf.keras.Model):
    def __init__(self, params):
        super().__init__()

        input_features = params["input_features"]
        n_units = params["n_units"]
        n_heads = params["n_heads"]
        n_labels = params["n_labels"]
        n_units_att = params["n_units_att"]

        self.embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_units),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation("relu"),
            ]
        )

        self.self_attention_modules = tf.keras.Sequential(
            [transformer.MultiHeadSelfAttentionBlock(n_units_att, n_heads)]
            * params["n_attention_layers"]
        )

        self.attention_pooling = transformer.MultiHeadAttentionPooling(
            1, n_units_att, n_heads
        )

        self.dense_output = tf.keras.layers.Dense(n_labels, activation="softmax")

    @tf.function
    def call(self, x):

        x = self.embeddings(x)
        x = self.self_attention_modules(x)
        x = self.attention_pooling(x)
        x = x[:, 0]
        x = self.dense_output(x)

        return x


class AttentionModule(tf.keras.layers.Layer):
    def __init__(self, n_units, n_heads):
        super().__init__()

        self.mha = transformer.MultiHeadSelfAttention(n_units, n_heads)
        self.norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.nn.elu

    def call(self, x):

        x0 = x
        x = self.mha(x)
        x = self.norm(x)
        x += x0
        x = self.activation(x)

        return x


class PoolingModel(tf.keras.Model):
    def __init__(self, params):
        super().__init__()

        input_features = params["input_features"]
        n_units = params["n_units"]
        n_heads = params["n_heads"]
        n_labels = params["n_labels"]

        self.embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_units),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation("elu"),
                tf.keras.layers.Dense(n_units * 3),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation("elu"),
            ]
        )

        self.max_pooling = tf.keras.layers.GlobalMaxPool1D()

        self.dense_output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("elu"),
                tf.keras.layers.Dense(n_labels, activation="softmax"),
            ]
        )

    @tf.function
    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]

        x = self.embeddings(x)
        x = self.max_pooling(x)
        x = self.dense_output(x, training=training)

        return x


def plot_batch(x, y=None):
    import matplotlib.pyplot as plt

    plt.scatter(x[0][:, 0], x[0][:, 1])
    if y is not None:
        plt.title(f"{y[0]}")
    plt.show()


if __name__ == "__main__":
    typer.run(main)
