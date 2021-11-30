import gym
from time import time
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('LunarLander-v2')


n_games = 1000

start = time()

agent = Agent(
    20000, env.observation_space.shape, env.action_space.n, 300)

scores = []
score_average = []


def get_fps(start):
    fps = 1 / (time() - start)
    return fps  # frames per second


for i in range(n_games):
    print(f'Game {i}')
    observation = env.reset()
    score = 0
    for t in range(500):
        # env.render()
        # action = env.action_space.sample()
        action = agent.choose_action(observation)
        obs_, reward, done, info = env.step(action)
        score += reward
        agent.store(observation, action, reward, obs_, done)
        observation = obs_
        # print(get_fps(start))
        start = time()
        if done:
            # print(f'Score: {score}')
            break
        scores.append(score)
        score_average.append(np.mean(scores[-100:]))
        if i > 990:
            env.render()
    agent.fit(1, 64)
    # print(f'Score: {score}')
    # break

plt.plot(agent.model.loss)
plt.title('Loss')
plt.show()

plt.plot(scores)
plt.plot(score_average)
plt.title('Scores')
plt.legend(['Scores', 'Average scores'])
plt.show()
