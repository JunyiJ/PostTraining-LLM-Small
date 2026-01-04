import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Computes the DPO loss for a batch of preference pairs.
    
    Args:
        policy_chosen_logps: Log probs of chosen responses from the model being trained.
        policy_rejected_logps: Log probs of rejected responses from the model being trained.
        ref_chosen_logps: Log probs of chosen responses from the frozen reference model.
        ref_rejected_logps: Log probs of rejected responses from the frozen reference model.
        beta: Temperature parameter (usually 0.1 to 0.5). Higher beta = strictly stays close to ref model.
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    logits = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)
    losses = -F.logsigmoid(beta * (logits))
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()
