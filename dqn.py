import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.activation import ReLU


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Net, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.fc1 = nn.Linear(input_size[0], hidden_size,
        #                      device=self.device, bias=True)
        # self.fc2 = nn.Linear(hidden_size, hidden_size,
        #                      device=self.device, bias=True)
        # self.fc3 = nn.Linear(hidden_size, int(hidden_size/2),
        #                      device=self.device, bias=True)
        # self.fc4 = nn.Linear(int(hidden_size/2), output_size,
        #                      device=self.device, bias=True)

        self.net = nn.Sequential(
            nn.Linear(input_size[0], hidden_size,
                      device=self.device, bias=True),

            nn.Linear(hidden_size, int(hidden_size/2),
                      device=self.device, bias=True),

            nn.Linear(int(hidden_size/2), output_size,
                      device=self.device, bias=True),

        ).to(self.device)

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.to(self.device)

        self.loss = []

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        x = self.net(x)
        # x = np.argmax(x.detach().cpu().numpy())
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = self.net(x)
        x = np.argmax(x.detach().cpu().numpy())
        return x

    def fit(self, x, y, epochs=1, batch_size=32, lr=0.001):
        x = np.array(x)
        y = np.array(y)
        running_loss = 0.0
        for epoch in range(epochs):
            inputs = torch.tensor(x, dtype=torch.float32, device=self.device)
            label = torch.tensor(
                y, dtype=torch.float32, device=self.device)
            # forwards
            y_pred = self.net(inputs)
            loss = self.criterion(y_pred, label)
            # backwards
            loss.backward()

            # update weights
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss.append(loss.item())
