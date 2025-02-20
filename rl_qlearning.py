import numpy as np
import pandas as pd
import yfinance as yf
import random
import gymnasium as gym
from gymnasium import Env, spaces
import streamlit as st
import plotly.graph_objects as go

# Set the random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

st.header('Stock RL with Q-Learning')
st.write('This is a basic Stock agent that uses Q-learning to determine optimal strategy to buy, hold and sell an individual stock'
         )
st.write('This app compares a random agent who chooses actions randomly, a buy and hold strategy and a Stock agent trained with Q-learning to pick the optimal strategy.')
st.write('The agent is trained on price data from 2014-2024 and tested on data from 2024-2025.')
st.write('')
ticker = st.selectbox('Select from the following', ('aapl', 'tsla', 'goog', 'msft', 'nflx', 'amzn'))
data = yf.download(ticker, '2014-01-01', '2025-01-01',multi_level_index=False)[['Close']]
train = data[:-252]
test = data[-252:]


class StockEnv(Env):
    def __init__(self, data, initial=10000, no_buckets=[10, 50, 5]):
        super().__init__()
        self.data = np.array(data)
        self.n_steps = len(data)
        self.initial = initial
        self.action_space = spaces.Discrete(3)  # hold, buy, sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.state_value_bounds = [
            (np.min(self.data), np.max(self.data)),
            (0, self.initial * 10),
            (0, 1000),
        ]
        self.no_buckets = no_buckets

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)  # Set NumPy's random seed
            random.seed(seed)  # Set Python's random seed
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial
        self.stock_held = 0
        self.done = False
        state = self._get_observation()
        return self._discretize_state(state), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called after the episode has finished.")

        current_price = self.data[self.current_step][0]
        if action == 1:  # Buy
            if self.cash >= current_price:
                num_shares = self.cash // current_price
                self.cash -= num_shares * current_price
                self.stock_held += num_shares
        elif action == 2:  # Sell
            if self.stock_held > 0:
                self.cash += self.stock_held * current_price
                self.stock_held = 0
        

        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.done = True

        total = self.cash + self.stock_held * current_price
        reward = total - self.initial
        state = self._get_observation()
        return state, reward, self.done, False, {}

    def _get_observation(self):
        current_price = self.data[self.current_step][0]
        return np.array([current_price, self.cash, self.stock_held], dtype=np.float32)

    def _discretize_state(self, state_value):
        bucket_indexes = []
        for i in range(len(state_value)):
            if state_value[i] <= self.state_value_bounds[i][0]:
                bucket_index = 0
            elif state_value[i] >= self.state_value_bounds[i][1]:
                bucket_index = self.no_buckets[i] - 1
            else:
                bound_width = self.state_value_bounds[i][1] - self.state_value_bounds[i][0]
                offset = (self.no_buckets[i] - 1) * self.state_value_bounds[i][0] / bound_width
                scaling = (self.no_buckets[i] - 1) / bound_width
                bucket_index = int(round(scaling * state_value[i] - offset))
            bucket_indexes.append(bucket_index)
        return tuple(bucket_indexes)


def random_s(data, seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    stock = StockEnv(data, 10000)
    state, _ = stock.reset(seed=seed)
    done = False
    rewards = []
    for _ in range(len(data) - 1):
        action = random.randint(0, 2)  # Random actions
        state, reward, _, _, _ = stock.step(action)
        rewards.append(reward)
    return np.array(rewards)


rand = np.array([random_s(test, seed=SEED) for _ in range(100)])
avg = np.mean(rand, axis=0)

stock = StockEnv(data, 10000)
state_size = np.prod(stock.no_buckets)
action_size = stock.action_space.n
q_table = np.zeros((stock.no_buckets + [action_size]))

def rl(data, epsilon, min_epsilon, eps_decay, lr, gamma, seed=SEED):
    stock.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

    
    state, _ = stock.reset(seed=seed)
    rewards = []

    for _ in range(len(data) - 1):
        if random.random() < epsilon:
            action = stock.action_space.sample()  # Exploration
        else:
            action = np.argmax(q_table[state])  # Exploitation

        new_state, reward, _, _, _ = stock.step(action)
        dis_new_state = stock._discretize_state(new_state)
        fut_q_value = np.amax(q_table[dis_new_state])
        current_q = q_table[state + (action,)]
        new_q = (1 - lr) * current_q + lr * (reward + gamma * fut_q_value)
        q_table[state + (action,)] = new_q
        state = dis_new_state
        rewards.append(reward)
        epsilon *= eps_decay
        epsilon = max(min_epsilon, epsilon)
    return rewards, q_table


def rl_test(data, q_table, seed=SEED):
    stock = StockEnv(data, 10000)
    stock.reset(seed=seed)
    state, _ = stock.reset()
    rewards = []
    history = []

    for i in range(len(data) - 1):
        action = np.argmax(q_table[state])  # Always pick the best action
        new_state, reward, _, _, _ = stock.step(action)
        dis_new_state = stock._discretize_state(new_state)
        state = dis_new_state
        rewards.append(reward)
        history.append(
            {
                "Date": data.index[i],
                "Price": data.values[i][0],
                "Shares": stock.stock_held,
                "Action": action,
                "Reward": reward,
            }
        )
    df = pd.DataFrame(history)
    return rewards, df
st.write('Choosing lower runs will result in faster testing, but possibly worse or no results.')

runs = st.slider('Pick how many times to run the agent',min_value=1,
                 max_value = 200, value = 10)

st.write('')
st.write('Leave default values if you do not understand what these parameters are.')
eps = st.slider('Select epsilon value', 0.0, 1.0, 1.0, 0.01)
min_eps = st.slider('Select minimum epsilon value', 0.0, 0.1, 0.1, 0.01)
eps_dec = st.slider('Select epsilon decay rate', 0.0, 0.99, 0.99, 0.01)
gam = st.slider('Select gamma value', 0.0, 1.0, 0.9, 0.01)
learn = st.slider('Select learning rate', 0.0, 1.0, 0.1, 0.01)

rews = []
q_tables = []

for _ in range(runs):
    rew, q_t = rl(train, eps, min_eps, eps_dec, learn, gam, seed=SEED)
    rews.append(rew[-1])
    q_tables.append(q_t)

max_idx = np.argmax(np.array(rews))
best_q = q_tables[max_idx]

rew_test, df = rl_test(test, best_q, seed=SEED)

tests = test.values
num_s = 10000 // tests[0]
bh_r = num_s[0] * tests - 10000

fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index, y=rew_test, name='rl'))
fig.add_trace(go.Scatter(x=test.index, y=avg, name='random'))
fig.add_trace(go.Scatter(x=test.index, y=bh_r.reshape(-1), name='buy and hold'))
fig.update_layout(hovermode='x', title=f'Profit on $10000 for {ticker}')
st.plotly_chart(fig)
st.write('Summary of the QL Agent')
st.write(df)
st.write('Reward is profit based on a $10000 portfolio.')
st.write('0 means hold, 1 is buy and 2 is sell.')
st.write('It is possible for the RL agent to produce 0 rewards. This means the agent was unable to learn in that run, and re-running should fix this.')
st.write('For someone who understands Q Learning this app can show the benefits of RL over buy and hold and the affect altering the parameters have.')
