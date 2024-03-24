from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from benchmark_utils.solver_class import CNN


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'CNN'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {}

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    def set_objective(self, train_loader):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(

                    nn.Conv2d(3, 6, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(6, 16, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),

                    nn.Flatten(),
                    nn.Linear(16 * 53 * 53, 120),
                    nn.ReLU(),
                    nn.Linear(120, 84),
                    nn.ReLU(),
                    nn.Linear(84, 10),
                    nn.ReLU(),
                    nn.Linear(10, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                x = self.network(x)
                return x

        net = Net()

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Running on MPS")

        net.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        clf = CNN(model=net,
                  criterion=criterion,
                  optimizer=optimizer,
                  train_loader=train_loader)

        self.clf = clf

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        clf = self.clf

        clf.fit(epochs=5)

    def get_next(self, n_iter):
        return n_iter + 1

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.clf, type='images')
