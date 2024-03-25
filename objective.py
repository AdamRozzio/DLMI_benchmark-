from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from benchmark_utils.processing import flatten_images


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "DLMI"

    # URL of the main repo for this benchmark.
    url = "https://github.com/#ORG/#BENCHMARK_NAME"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {}

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = []

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, train_dataset, train_loader, test_dataset, test_loader):

        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        return dict(train_dataset=train_dataset,
                    train_loader=train_loader,
                    test_dataset=test_dataset,
                    test_loader=test_loader
                    )

    def evaluate_result(self, model, type):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        if type == 'flatten':
            self.X_test = flatten_images(self.X_test)
            self.X_train = flatten_images(self.X_train)

        if type == 'images':
            self.X_train = self.train_dataset.X
            self.X_test = self.test_dataset.X
            self.y_train = self.train_dataset.y
            self.y_test = self.test_dataset.y
#TODO
        for data in self.X_train :
            inputs = data
            outputs = self.model(inputs)

        y_pred_train = model.predict(self.train_loader)
        y_pred_test = model.predict(self.test_loader)

        print("les prédictions de test sont", y_pred_test)
        print("les prédictions de train sont", y_pred_train)

        score_test = balanced_accuracy_score(self.y_test.cpu(),
                                             y_pred_test)
        score_train = balanced_accuracy_score(self.y_train.cpu(),
                                              y_pred_train)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            score_test=score_test,
            score_train=score_train,
            value=1-score_test
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(beta=np.zeros(self.X_test.shape[1]))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.

        return dict(train_loader=self.train_loader)