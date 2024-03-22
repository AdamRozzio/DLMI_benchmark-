from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.load_data import load_data, load_X_y
    from benchmark_utils.load_data import CustomDataset
    from torch.utils.data import DataLoader


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "simulated"

    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        csv_path_train = "dataset/dataDLMI-main/trainset/trainset_true.csv"
        images_path_train = "dataset/dataDLMI-main/trainset/"
        csv_path_test = "dataset/dataDLMI-main/testset/testset_data.csv"
        images_path_test = "dataset/dataDLMI-main/testset/"

        data_train = load_data(csv_path_train, images_path_train)
        data_test = load_data(csv_path_test, images_path_test)
        # The dictionary defines the keyword arguments for `Objective.set_data`

        X_train, y_train = load_X_y(data_train)
        X_test, y_test = load_X_y(data_test)
        print("après load X_y", X_train.shape)

        batch_size = 4

        train_dataset = CustomDataset(X_train, y_train, transform=None)
        test_dataset = CustomDataset(X_test, y_test, transform=None)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0)

        return dict(train_dataset=train_dataset,
                    train_loader=train_loader,
                    test_dataset=test_dataset,
                    test_loader=test_loader)
