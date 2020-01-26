import typer
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
import cytoolz as cz
from sklearn.preprocessing import MinMaxScaler

from .transformer import MultiHeadSelfAttention

np.random.seed(42)


def main(
    data_dir: Path = Path("data"),
    params_path: Path = Path("training/params.yml"),
    dataset_version: str = "100",
):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    X_train = np.load(data_dir / f"X_train_{dataset_version}.npy")
    y_train = np.load(data_dir / f"y_train.npy")
    X_test = np.load(data_dir / f"X_test_{dataset_version}.npy")
    y_test = np.load(data_dir / f"y_test.npy")

    preprocessor = MinMaxScaler(feature_range=(-1, 1))

    X_train, y_train = preprocess(X_train, y_train, params, preprocessor, mode="train")
    X_test, y_test = preprocess(X_test, y_test, params, preprocessor, mode="test")

    model = Model(params)

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
    # y = y.astype(np.long)

    # X = X[: params["batch_size"] * 10]
    # y = y[: params["batch_size"] * 10]

    return X, y


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Model(tf.keras.Model):
    def __init__(self, params):
        super().__init__()

        input_features = params["input_features"]
        n_units = params["n_units"]
        n_heads = params["n_heads"]
        n_labels = params["n_labels"]

        self.embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_units),
                tf.keras.layers.Activation("elu"),
                tf.keras.layers.Dense(n_units),
                tf.keras.layers.Activation("elu"),
            ]
        )

        # self.dense_up = tf.keras.layers.Dense(n_units * n_heads)

        self.self_attention_modules = tf.keras.Sequential(
            [AttentionModule(n_units, n_heads)] * 3
        )

        self.dense_output = tf.keras.layers.Dense(n_labels, activation="softmax")

    @tf.function
    def call(self, x):
        batch_size = tf.shape(x)[0]

        # print(x.shape)
        token = tf.constant([-1.0, 1.0, -1.0])[None, None, :]
        token = tf.tile(token, (batch_size, 1, 1))

        x = tf.concat([token, x], axis=1)
        # print(x.shape)

        # plot_batch(x.numpy())

        x = self.embeddings(x)

        # print(x.shape)

        # token = self.classifier_token.repeat(batch_size, 1, 1)

        # print(x.shape)

        # x = self.dense_up(x)

        # print(x.shape)

        x = self.self_attention_modules(x)

        print(x.shape)

        x = x[:, 0]

        # print(x.shape)

        x = self.dense_output(x)

        # print(x[0])

        # print(x.shape)

        return x


class AttentionModule(tf.keras.layers.Layer):
    def __init__(self, n_units, n_heads):
        super().__init__()

        self.mha = MultiHeadSelfAttention(n_units, n_heads)
        self.norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.nn.relu

    def call(self, x):

        x0 = x
        x = self.mha(x)
        x = self.norm(x)
        x += x0
        x = self.activation(x)

        return x


def plot_batch(x, y=None):
    import matplotlib.pyplot as plt

    plt.scatter(x[0][:, 0], x[0][:, 1])
    if y is not None:
        plt.title(f"{y[0]}")
    plt.show()


if __name__ == "__main__":
    typer.run(main)
