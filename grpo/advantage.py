def compute_advantage(rewards):
    mean_reward = sum(rewards) * 1.0 / len(rewards)
    advantages = [r - mean_reward for r in rewards]
    return advantages