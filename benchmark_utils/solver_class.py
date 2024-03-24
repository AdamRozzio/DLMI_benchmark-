import torch
import time


class ResNet:
    def __init__(self, model, criterion, optimizer, train_loader, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device

    def fit(self, epochs=1):
        since = time.time()
        self.model.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
            running_loss = 0.0
            for (i, data) in enumerate(self.train_loader, 0):
                inputs, labels = data['image'], data['label']
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() 
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}]',
                          f'loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        time_elapsed = time.time() - since
        print('Finished Training')
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


    def fit(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
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