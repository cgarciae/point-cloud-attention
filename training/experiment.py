import typer
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import cytoolz as cz
from sklearn.preprocessing import MinMaxScaler

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

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    ds_test = torch.utils.data.TensorDataset(X_test, y_test)

    model = Model2(params)

    print(f"Parameters: {count_parameters(model)}")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(params, ds_train, ds_test, loss_fn, optimizer)

    trainer.fit(
        model, params["epochs"], params["steps_per_epoch"], params["validation_steps"]
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
    y = y.astype(np.long)

    # X = X[: params["batch_size"] * 10]
    # y = y[: params["batch_size"] * 10]

    return torch.tensor(X), torch.tensor(y)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Model(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        input_features = params["input_features"]
        n_units = params["n_units"]
        n_heads = params["n_heads"]
        n_labels = params["n_labels"]

        self.embeddings = torch.nn.Sequential(
            torch.nn.Linear(input_features, n_units),
            torch.nn.ELU(),
            torch.nn.Linear(n_units, n_units),
            torch.nn.ELU(),
        )

        self.classifier_token = torch.nn.parameter.Parameter(torch.rand(1, 1, n_units))
        self.classifier_token.data.uniform_(0.0, 1.0)

        self.self_attention_modules = [
            torch.nn.MultiheadAttention(n_units, n_heads),
            torch.nn.MultiheadAttention(n_units, n_heads),
            torch.nn.MultiheadAttention(n_units, n_heads),
        ]

        self.dense_output = torch.nn.Linear(n_units, n_labels)

    def forward(self, x):
        batch_size = x.shape[0]

        # print(x.shape)
        token = torch.tensor([-1.0, 1.0, -1.0])[None, None, :].repeat(batch_size, 1, 1)

        x = torch.cat((token, x), 1)

        x = self.embeddings(x)

        # print(x.shape)

        # token = self.classifier_token.repeat(batch_size, 1, 1)

        # print(x.shape)

        for module in self.self_attention_modules:
            x, _ = module(x, x, x)
            x = F.relu(x)
            # print(x.shape)

        x = x[:, 0]

        # print(x.shape)

        x = self.dense_output(x)

        # print(x[0])

        # print(x.shape)

        return x


class Model2(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        input_features = params["input_features"] * 100
        n_units = params["n_units"]
        n_heads = params["n_heads"]
        n_labels = params["n_labels"]

        self.embeddings = torch.nn.Sequential(
            torch.nn.Linear(input_features, n_units),
            torch.nn.ELU(),
            torch.nn.Linear(n_units, n_units),
            torch.nn.ELU(),
        )

        self.dense_output = torch.nn.Linear(n_units, n_labels)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(x.shape)

        x = x.view(batch_size, -1)
        # print(x.shape)

        x = self.embeddings(x)
        # print(x.shape)

        x = self.dense_output(x)
        # print(x.shape)

        return x


class Trainer:
    def __init__(self, params, ds_train, ds_test, loss_fn, optimizer):
        self.params = params
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def fit(self, model, epochs, steps_per_epoch, validation_steps):

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            running_acc = 0.0

            trainloader = self.get_loader(self.ds_train)
            trainloader = cz.take(steps_per_epoch, trainloader)

            for step, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # plot_batch(inputs.numpy(), labels.numpy())

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                loss = self.loss_fn(outputs, labels)
                acc = (outputs.argmax(1) == labels).float().mean()

                loss.backward()

                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_acc += acc.item()

            print(
                f"[{epoch}] loss: {running_loss / (step + 1)}, acc: {running_acc / (step + 1)}"
            )
            running_loss = 0.0
            running_acc = 0.0

    def get_loader(self, ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=self.params["batch_size"], shuffle=True, drop_last=True
        )


def plot_batch(x, y):
    import matplotlib.pyplot as plt

    plt.scatter(x[0][:, 0], x[0][:, 1])
    plt.title(f"{y[0]}")
    plt.show()


if __name__ == "__main__":
    typer.run(main)
