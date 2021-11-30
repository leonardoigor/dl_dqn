import numpy as np


class ReplayMemory:
    def __init__(self, capacity,  input_size, num_actions):

        self.capacity = capacity
        self.input_size = input_size
        self.num_actions = num_actions
        self.action_space = [i for i in range(num_actions)]

        self.state = np.zeros((capacity,)+input_size)
        self.reward_state = np.zeros(capacity)
        self.next_state = np.zeros((capacity,)+input_size)
        self.done = np.zeros(capacity)

        self.action_state = np.zeros((capacity, num_actions))
        self.current_index = 0

    def store_transition(self, state, action, reward, next_state, done):
        index = self.current_index % self.capacity
        self.state[index] = state
        self.action_state[index, action] = 1
        self.reward_state[index] = reward
        self.next_state[index] = next_state
        self.done[index] = done
        self.current_index += 1

        if self.current_index > self.capacity:
            self.current_index = 0

    def sample(self, batch_size):
        max_mem = min(self.current_index, self.capacity)
        batch_index = np.random.choice(max_mem, batch_size)

        states = self.state[batch_index]
        actions = self.action_state[batch_index]
        rewards = self.reward_state[batch_index]
        states_next = self.next_state[batch_index]
        dones = self.done[batch_index]

        return states, actions, rewards, states_next, dones
