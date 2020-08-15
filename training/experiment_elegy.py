import os
import time
import typing as tp
from pathlib import Path

import cytoolz as cz
import dataget
import jax
import jax.numpy as jnp
import numpy as np
from numpy.core.fromnumeric import squeeze
import tensorflow as tf
import typer
import yaml
from jax.experimental import optix
from sklearn.preprocessing import MinMaxScaler

import elegy
from elegy.nn.multi_head_attention import MultiHeadAttention

from . import set_transformer

np.random.seed(42)


def main(
    data_dir: Path = Path("data"),
    params_path: Path = Path("training/params.yml"),
    train_version: str = "max",
    test_version: str = "max",
    model_type: str = "attention",
    gpu_off: bool = False,
):
    if gpu_off:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model_dir = f"models/{train_version}_{test_version}/{int(time.time())}"

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    df_train, df_test = dataget.kaggle("cristiangarcia/pointcloudmnist2d").get(
        files=["train.csv", "test.csv"]
    )

    X_train = df_train[df_train.columns[1:]].to_numpy()
    y_train = df_train[df_train.columns[0]].to_numpy()
    X_test = df_test[df_test.columns[1:]].to_numpy()
    y_test = df_test[df_test.columns[0]].to_numpy()

    X_train = X_train.reshape(len(X_train), -1, 3)
    X_test = X_test.reshape(len(X_test), -1, 3)

    print(X_train.shape)
    print(X_test.shape)

    preprocessor = MinMaxScaler(feature_range=(-1, 1))

    X_train, y_train = preprocess(X_train, y_train, params, preprocessor, mode="train")
    X_test, y_test = preprocess(X_test, y_test, params, preprocessor, mode="test")

    if model_type == "attention":
        module = Attention(
            n_labels=params["n_labels"],
            n_units=params["n_units"],
            n_units_att=params["n_units_att"],
            n_heads=params["n_heads"],
            n_layers=params["n_attention_layers"],
            activation=jax.nn.relu,
        )
    elif model_type == "pooling":
        module = DeepSet(params, activation=jax.nn.relu)
    else:
        raise ValueError(f"Unknown model_type type: {model_type}")

    model = elegy.Model(
        module,
        loss=elegy.losses.SparseCategoricalCrossentropy(),
        metrics=[elegy.metrics.SparseCategoricalAccuracy()],
        optimizer=optix.adam(params["lr"]),
    )

    model.summary(X_train[:2], depth=1)
    plot_batch(X_train, y_train)

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        steps_per_epoch=params["steps_per_epoch"],
        validation_data=(X_test, y_test),
        validation_steps=params["validation_steps"],
        callbacks=[
            # tf.keras.callbacks.LearningRateScheduler(
            #     tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            #         [500, 1200], [0.001, 0.0005, 0.0001]
            #     )
            # ),
            elegy.callbacks.ModelCheckpoint(
                path=f"{model_dir}/saved_model", save_best_only=True
            ),
            elegy.callbacks.TensorBoard(logdir=model_dir),
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


class Attention(elegy.nn.Sequential):
    def __init__(
        self,
        n_labels,
        n_units,
        n_units_att,
        n_heads,
        n_layers,
        activation: tp.Callable = jax.nn.relu,
        **kwargs,
    ):
        def get_first(x):
            return x[:, 0]

        super().__init__(
            lambda: [
                elegy.nn.Linear(n_units),
                elegy.nn.LayerNormalization(),
                activation,
                elegy.nn.Sequential(
                    lambda: [
                        set_transformer.IMAB(
                            n_units_att,
                            n_heads,
                            inducing_points=20,
                            activation=activation,
                        )
                        for _ in range(n_layers)
                    ],
                    name="transformer_encoder",
                ),
                set_transformer.PMA(
                    n_labels, n_units_att, n_heads, activation=activation
                ),
                set_transformer.MAB(n_units_att, n_heads, activation=activation),
                elegy.nn.Linear(1),
                jnp.squeeze,
                jax.nn.softmax,
            ]
        )


class TransformerEncoder(elegy.nn.Sequential):
    def __init__(
        self, n_units, n_heads, n_layers, activation: tp.Callable = jax.nn.elu, **kwargs
    ):
        super().__init__(
            lambda: [
                TransfomerEncoderModule(n_units, n_heads, activation)
                for _ in range(n_layers)
            ],
            **kwargs,
        )

    @tf.function
    def call(self, x):

        x = self.embeddings(x)
        x = self.self_attention_modules(x)
        x = self.attention_pooling(x)
        x = x[:, 0]
        x = self.dense_output(x)

        return x


class TransfomerEncoderModule(elegy.Module):
    def __init__(self, n_units, n_heads, activation: tp.Callable = jax.nn.elu):
        super().__init__()
        self.n_units = n_units
        self.n_heads = n_heads
        self.activation = activation

    def call(self, x):
        output_size = x.shape[-1]

        x = x + MultiHeadAttention(self.n_units, self.n_heads, output_size=output_size)(
            x
        )
        x0 = x
        x = elegy.nn.LayerNormalization()(x)
        x = elegy.nn.Linear(output_size)(x)
        x = x0 + self.activation(x)
        x = elegy.nn.LayerNormalization()(x)

        return x


class DeepSet(elegy.Module):
    def __init__(self, params, activation: tp.Callable = jax.nn.relu):
        super().__init__()

        self.input_features = params["input_features"]
        self.n_units = params["n_units"]
        self.n_heads = params["n_heads"]
        self.n_labels = params["n_labels"]
        self.activation = activation

    def call(self, x):
        def global_maxpooling_1d(x):
            return jnp.max(x, axis=1)

        return elegy.nn.sequential(
            elegy.nn.Linear(self.n_units),
            elegy.nn.LayerNormalization(),
            self.activation,
            elegy.nn.Linear(self.n_units * 3),
            elegy.nn.LayerNormalization(),
            self.activation,
            global_maxpooling_1d,
            elegy.nn.Linear(self.n_units),
            elegy.nn.BatchNormalization(),
            self.activation,
            elegy.nn.Linear(self.n_labels),
            jax.nn.softmax,
        )(x)


def plot_batch(x, y=None):
    import matplotlib.pyplot as plt

    plt.scatter(x[0][:, 0], x[0][:, 1])
    if y is not None:
        plt.title(f"{y[0]}")
    plt.show()


if __name__ == "__main__":
    typer.run(main)
