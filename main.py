from dqn import DQNTrainer

if __name__ == '__main__':
    trainer = DQNTrainer(500,200, 5)
    trainer.train()