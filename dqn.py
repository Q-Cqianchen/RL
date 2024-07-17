import torch
from torch import nn, optim
import torch.functional as F
import gym
import random
from utils import ReplayBuffer


# 输入为state，输出对于每个actions的Q值的预测：Q(s,a) for all a
class DQNModel(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_size):
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.fcs(x)



class DQNTrainer(object):
    def __init__(self, episodes, steps, target_update_episodes=5, batch_size=64, gamma=0.99):
        '''
        :param episodes: 训练几个回合
        :param steps: 每回合至多多少steps
        :param target_update_episodes: 每多少回合更新一次目标网络
        :param batch_size: 批量大小
        :param gamma: 折扣率
        '''
        self.env = gym.make("CartPole-v1")
        self.action_dim, self.state_dim = self._get_info()
        self.replay_buffer = ReplayBuffer(capacity=2000)
        self.episodes = episodes
        self.steps = steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_episodes = target_update_episodes

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_model = DQNModel(self.action_dim, self.state_dim, 32).to(self.device)
        self.target_q_model = DQNModel(self.action_dim, self.state_dim, 32).to(self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())

    # 获取环境的动作、状态空间大小
    def _get_info(self, ):
        action_dim = self.env.action_space.n  # 2
        state_dim = self.env.observation_space.shape[0]  # (4,)[0]=4
        print('action_dim', action_dim)
        print('state_dim', state_dim)
        return action_dim, state_dim

    # epsilon-greedy选择动作
    def _select_action(self, state, epsilon):
        if random.random() < epsilon:  # 在当前状态下随机选择动作
            action = random.randint(0, self.action_dim-1)
        else:  # 选择当前状态下q值最大的动作
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            dist = self.q_model(state)[0]
            action = dist.argmax().item()
        return action

    def _save_model(self, checkpoint_name):
        torch.save(self.target_q_model, checkpoint_name)

    def train(self, ):
        optimizer = optim.Adam(self.q_model.parameters())
        loss_fn = nn.MSELoss()
        for episode in range(self.episodes):  # 一个回合
            state, _ = self.env.reset()
            epsilon = max(1 - episode / 500, 0.01)  # 随着训练的进行而逐渐减小至0.01
            for step in range(self.steps):  # 每个回合至多200steps
                reached = step
                action = self._select_action(state, epsilon)
                next_state, reward, done, truncation, info = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

                # 该回合是否结束
                if done:
                    break
                # 只要replay_buffer大于batch_size，则训练q_model
                if self.replay_buffer.size() < self.batch_size:
                    continue

                experiences = self.replay_buffer.sample(self.batch_size)
                states, actions, rewards, next_states, dones = zip(*experiences)

                states = torch.tensor(states, dtype=torch.float32, device=self.device)  # (batch_size, state_dim)
                actions = torch.tensor(actions, dtype=torch.int64, device=self.device)  # (batch_size,)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # (batch_size,)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
                dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

                q_values = self.q_model(states).gather(dim=1, index=actions.unsqueeze(1))  # 为每个状态实际选择的动作预测Q值 (batch_size, 1)
                next_q_values = self.target_q_model(next_states).max(dim=1)[0]  # .max()会返回最大值及其索引 (batch_size,)
                expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)  # (batch_size,)

                # 注意，由于目的是优化q_values(其来自于q_model)，为避免梯度流传经target_q_model，需将expected_q_values从计算图上拆卸下来
                loss = loss_fn(q_values, expected_q_values.unsqueeze(1).detach())
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪(可选)
                for param in self.q_model.parameters():
                    param.data.clamp_(-1, 1)
                optimizer.step()

            print(f"Episode {episode+1} reach {reached}")
            # 定期把在线模型的权重复制到目标模型中，维持目标模型的稳定
            if (episode + 1) % self.target_update_episodes == 0:
                self.target_q_model.load_state_dict(self.q_model.state_dict())

        self._save_model("models/DQN/dqn.pdparams")