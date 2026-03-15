import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import MedicalNet
from data import load_hospital_data
from opacus import PrivacyEngine
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hospital_id = int(sys.argv[1])
trainloader = load_hospital_data(hospital_id)

model = MedicalNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Differential Privacy
privacy_engine = PrivacyEngine()
model, optimizer, trainloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=trainloader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        model.train()
        for epoch in range(2):
            for X, y in trainloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"Hospital {hospital_id} Privacy ε: {epsilon}")

        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in trainloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        return float(1 - correct / total), len(trainloader.dataset), {"accuracy": correct / total}

fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())