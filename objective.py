from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score


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

    def set_data(self,
                 train_dataset,
                 train_loader,
                 test_dataset,
                 test_loader,
                 data_train_bio):

        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.data_train_bio = data_train_bio

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

        y_train = self.train_dataset.y
        y_test = self.test_dataset.y

        y_pred_train = model.predict(self.train_loader)
        y_pred_test = model.predict(self.test_loader)

        score_train = balanced_accuracy_score(y_pred_train, y_train)
        print("score_train", score_train)

        score_test = balanced_accuracy_score(y_pred_test, y_test)
        print("score_test", score_test)

        for i in range(len(y_train)):
            if i % 17 == 1:
                print('truth', 'pred', (y_train[i], y_pred_train[i]))

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            bas_test=score_test,
            bas_train=score_train,
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

        return dict(train_loader=self.train_loader,
                    data_train_bio=self.data_train_bio)
