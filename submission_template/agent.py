import numpy as np


class Agent:
    def __init__(self):
        """
        Load your trained model here.
        This is called once before any matches begin.
        Keep it fast — don't do training here.
        """
        # Example (commented): load a Stable-Baselines3 SAC checkpoint.
        # from stable_baselines3 import SAC
        # self.model = SAC.load("my_agent.zip")

        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Given an observation, return an action.

        Args:
            observation: numpy array of shape (18,)
                [0:2]   player 1 position (x, y)
                [2]     player 1 angle
                [3:5]   player 1 velocity (vx, vy)
                [5]     player 1 angular velocity
                [6:8]   player 2 position (x, y)
                [8]     player 2 angle
                [9:11]  player 2 velocity (vx, vy)
                [11]    player 2 angular velocity
                [12:14] puck position (x, y)
                [14:16] puck velocity (vx, vy)
                [16]    has puck (player 1, timer)
                [17]    has puck (player 2, timer)
        Returns:
            action: numpy array of shape (4,) — values in [-1, 1]
                [0] force x
                [1] force y
                [2] torque
                [3] shoot (>0.5 triggers a shot)
        """
        # Replace this random policy with your trained model, e.g.:
        # action, _ = self.model.predict(observation, deterministic=True)
        # return np.asarray(action, dtype=np.float32)

        return np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)


if __name__ == "__main__":
    agent = Agent()
    action = agent.act(np.zeros(18, dtype=np.float32))
    print(f"action shape: {action.shape}")
