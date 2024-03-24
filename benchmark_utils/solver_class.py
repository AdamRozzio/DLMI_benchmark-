import torch


class CNN:
    def __init__(self, model, criterion, optimizer, train_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader

    def fit(self, epochs=2):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data['image'], data['label']
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}]',
                          f'loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print('Finished Training')

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                inputs = data['image']
                outputs = self.model(inputs)
                binary_predictions = (outputs >= 0.5).float()
                predictions.extend(binary_predictions.cpu().numpy().tolist())
        return predictions
