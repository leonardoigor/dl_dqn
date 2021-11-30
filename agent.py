from memory import ReplayMemory
from dqn import Net
import numpy as np
import torch


class Agent:
    def __init__(self, capacity,  input_size, output_size, hidden_size):
        self.capacity = capacity
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.memory = ReplayMemory(capacity, input_size, output_size)
        self.model = Net(input_size, output_size, hidden_size)
        self.gamma = 0.99
        self.epsilon = 1.0

    def forward(self, state):
        return self.model.forward(state)

    def predict(self, state):
        return self.model.predict(state)

    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.memory.action_space)
        else:
            action = self.predict(state)

        return action

    def fit(self,  epochs=1, batch_size=32):
        if self.memory.current_index < batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample(
            batch_size)

        actions_values = self.memory.action_space
        actions_index = np.dot(actions, actions_values)

        q_next = self.model.forward(next_states)
        q_eval = self.model.forward(next_states)
        q_pred = self.model.forward(states)
        q_target = q_pred
        max_actions = np.argmax(q_eval.detach().numpy(), axis=1)

        batch_index = np.arange(batch_size, dtype=np.int32)
        q_next_np = q_next.detach().numpy()
        calc = rewards+self.gamma * \
            q_next_np[batch_index, max_actions]*(1-dones)
        try:
            q_target[batch_index, actions_index] = torch.tensor(
                calc, dtype=torch.float32)

        except Exception as e:
            print('error')
            pass

        self.model.fit(states, q_target.detach().numpy(), epochs)
        self.epsilon = max(self.epsilon*0.99, 0.01)

    def sample(self, batch_size):

        return self.memory.sample(batch_size)

    def store(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
