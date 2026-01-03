import pytest
import torch
import torch.nn.functional as F

from dpo.dpo_loss import dpo_loss


def test_dpo_loss_matches_manual():
    policy_chosen = torch.tensor([0.2, -0.1])
    policy_rejected = torch.tensor([-0.4, 0.3])
    ref_chosen = torch.tensor([0.0, 0.0])
    ref_rejected = torch.tensor([0.1, -0.2])
    beta = 0.1

    loss_sum, chosen_r_sum, rejected_r_sum = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta
    )

    logits = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    expected_loss = (-F.logsigmoid(beta * logits)).sum()
    expected_chosen = (beta * (policy_chosen - ref_chosen)).sum()
    expected_rejected = (beta * (policy_rejected - ref_rejected)).sum()

    assert torch.allclose(loss_sum, expected_loss, atol=1e-6)
    assert torch.allclose(chosen_r_sum, expected_chosen, atol=1e-6)
    assert torch.allclose(rejected_r_sum, expected_rejected, atol=1e-6)


def test_dpo_rewards_detached_and_loss_has_grads():
    policy_chosen = torch.tensor([0.5, -0.2], requires_grad=True)
    policy_rejected = torch.tensor([-0.1, 0.4], requires_grad=True)
    ref_chosen = torch.tensor([0.0, 0.0], requires_grad=True)
    ref_rejected = torch.tensor([0.0, 0.0], requires_grad=True)

    loss_sum, chosen_r, rejected_r = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.2
    )

    assert chosen_r.requires_grad is False
    assert rejected_r.requires_grad is False

    loss_sum.backward()
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None
