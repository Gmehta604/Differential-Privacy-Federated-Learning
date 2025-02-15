import fdoclwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from src.models.model import SimpleNN
from src.utils.privacy_utils import add_noise

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, optimizer, epsilon):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.epsilon = epsilon  # Privacy budget

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Training loop with differential privacy
        for epoch in range(5):  # Local epochs
            for batch_idx, (data, target) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                
                # Add noise for differential privacy
                add_noise(self.model, self.epsilon)
                
                self.optimizer.step()

        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.valloader:
                output = self.model(data)
                loss += nn.functional.cross_entropy(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        accuracy = correct / len(self.valloader.dataset)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}