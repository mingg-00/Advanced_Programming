"""
Reinforcement learning helpers built on stable-baselines3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)


ACTION_BUY = 1
ACTION_SELL = 2


@dataclass
class TradingEnvConfig:
    initial_cash: float = 10_000_000  # KRW
    transaction_fee: float = 0.00015
    tax_rate: float = 0.0025
    window_size: int = 30


class TradingEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, config: TradingEnvConfig | None = None):
        if config is None:
            config = TradingEnvConfig()

        self.config = config
        self.data = data
        self.prices = data["close"].values
        self.features = data[["open", "high", "low", "close", "volume"]].values
        self.current_step = config.window_size
        self.cash = config.initial_cash
        self.shares = 0
        self._last_price = self.prices[self.current_step - 1]

        obs_shape = (config.window_size, self.features.shape[1])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # hold, buy, sell

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.config.window_size
        self.cash = self.config.initial_cash
        self.shares = 0
        self._last_price = self.prices[self.current_step - 1]
        return self._get_observation(), {}

    def step(self, action: int):
        done = self.current_step >= len(self.prices) - 1

        price = self.prices[self.current_step]

        if action == ACTION_BUY:
            self._buy(price)
        elif action == ACTION_SELL:
            self._sell(price)

        reward = self._get_total_value(price) - self._get_total_value(self._last_price)
        reward /= self.config.initial_cash
        self._last_price = price

        self.current_step += 1
        done = done or self.current_step >= len(self.prices) - 1
        observation = self._get_observation()

        info = {"cash": self.cash, "shares": self.shares, "price": price}
        return observation, reward, done, False, info

    def _get_total_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.config.window_size
        end = self.current_step
        window = self.features[start:end]
        return window.astype(np.float32)

    def _buy(self, price: float) -> None:
        max_shares = int(self.cash / price)
        if max_shares <= 0:
            return
        cost = max_shares * price
        fee = cost * self.config.transaction_fee
        self.cash -= cost + fee
        self.shares += max_shares

    def _sell(self, price: float) -> None:
        if self.shares <= 0:
            return
        proceeds = self.shares * price
        fee = proceeds * self.config.transaction_fee
        tax = proceeds * self.config.tax_rate
        self.cash += proceeds - fee - tax
        self.shares = 0


def train_model(data: pd.DataFrame, timesteps: int = 10_000) -> PPO:
    env = TradingEnv(data)
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=0)
    logger.info("Training PPO model for %s timesteps", timesteps)
    model.learn(total_timesteps=timesteps)
    return model


def backtest(model: PPO, data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    env = TradingEnv(data)
    obs, _ = env.reset()
    rewards = []
    values = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
        rewards.append(reward)
        values.append(env._get_total_value(info["price"]))  # pylint: disable=protected-access

    result = pd.DataFrame(
        {"date": data.index[env.config.window_size : env.current_step], "value": values}
    ).set_index("date")
    total_return = (values[-1] / env.config.initial_cash) - 1 if values else 0.0
    sharpe_like = np.mean(rewards) / (np.std(rewards) + 1e-8)
    result["returns"] = result["value"].pct_change().fillna(0)
    result["cum_returns"] = (1 + result["returns"]).cumprod() - 1
    result["reward"] = rewards[: len(result)]
    return result, sharpe_like


