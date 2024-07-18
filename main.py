from dqn import DQNTrainer
from ddqn import DDQNTrainer

if __name__ == '__main__':
    # trainer = DQNTrainer(500,200, 5)
    trainer = DDQNTrainer(500, 200, 5)
    trainer.train()