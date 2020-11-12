import os
import time
import typing as tp
from pathlib import Path

import cytoolz as cz
import dataget
import elegy
from elegy.nn import transformers
import jax
import jax.numpy as jnp
import numpy as np
import typer
import yaml
from numpy.core.fromnumeric import squeeze
from sklearn.preprocessing import MinMaxScaler
import optax


np.random.seed(42)


def main(
    data_dir: Path = Path("data"),
    params_path: Path = Path("training/params.yml"),
    train_version: str = "max",
    test_version: str = "max",
    model_type: str = "attention",
    gpu_off: bool = False,
    debug: bool = False,
):

    if debug:
        import debugpy

        print("Waiting debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

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

    module = PointCloudTransformer(
        n_labels=params["n_labels"],
        n_units=params["n_units"],
        n_units_att=params["n_units_att"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        activation=jax.nn.relu,
    )

    model = elegy.Model(
        module,
        loss=elegy.losses.SparseCategoricalCrossentropy(),
        metrics=[elegy.metrics.SparseCategoricalAccuracy()],
        optimizer=optax.adam(params["lr"]),
        # run_eagerly=True,
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
        # callbacks=[
        #     elegy.callbacks.ModelCheckpoint(
        #         path=f"{model_dir}/saved_model", save_best_only=True
        #     ),
        #     elegy.callbacks.TensorBoard(logdir=model_dir),
        # ],
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


class PointCloudTransformer(elegy.Module):
    def __init__(
        self,
        n_labels: int,
        n_units: int,
        n_units_att: int,
        n_heads: int,
        n_layers: int,
        activation: tp.Callable,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.n_units = n_units
        self.n_units_att = n_units_att
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.activation = activation

    def call(self, x):

        queries = self.add_parameter(
            "queries",
            shape=[1, self.n_labels, self.n_units],
            initializer=elegy.initializers.VarianceScaling(),
        )
        queries = jnp.broadcast_to(queries, shape=x.shape[0:1] + queries.shape[1:])

        x = elegy.nn.Linear(self.n_units)(x)
        x = elegy.nn.LayerNormalization()(x)

        x = transformers.Transformer(
            head_size=self.n_units_att,
            num_heads=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            output_size=self.n_units,
            activation=self.activation,
        )(x, queries)

        x = elegy.nn.Linear(1)(x)[..., 0]
        x = jax.nn.softmax(x)

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
