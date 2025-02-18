"""Tests for bdpy.recon.torch.modules.optimizer"""

from __future__ import annotations

import unittest

from functools import partial
import numpy as np
import torch.nn as nn
import torch.optim as optim
from bdpy.recon.torch.modules import build_generator, ArbitraryLatent
from bdpy.recon.torch.modules import build_optimizer_factory, build_scheduler_factory


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class TestBuildOptimizerFactory(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.optimizer.build_optimizer_factory"""

    def test_build_optimizer_factory(self):
        generator = build_generator(MLP(64, 10))
        latent = ArbitraryLatent(
            (1, 64), init_fn=partial(nn.init.normal_, mean=0, std=1)
        )
        optimizer_factory = build_optimizer_factory(optim.SGD, lr=0.1)
        optimizer = optimizer_factory(generator, latent)
        self.assertIsInstance(
            optimizer,
            optim.SGD,
            msg="optimizer_factory should return an instance of optim.Optimizer",
        )

        latent.reset_states()
        generator.reset_states()
        latent_prev = latent().detach().clone().numpy()
        optimizer.zero_grad()
        output = generator(latent())
        loss = output.sum()
        loss.backward()
        latent_next_expected = (
            latent_prev - 0.1 * latent().grad.detach().clone().numpy()
        )
        optimizer.step()
        latent_next = latent().detach().clone().numpy()
        np.testing.assert_allclose(
            latent_next,
            latent_next_expected,
            rtol=1e-6,
            err_msg="Optimizer does not update the latent variable correctly.",
        )

        # check if all the frozen generator's gradients are None
        generator_grad = [p.grad for p in generator.parameters()]
        self.assertTrue(
            all([g is None for g in generator_grad]),
            msg="Frozen generator's gradients should be None after the optimizer step.",
        )


class TestBuildSchedulerFactory(unittest.TestCase):
    """Tests for bdpy.recon.torch.modules.optimizer.build_scheduler_factory"""

    def test_build_scheduler_factory(self):
        generator = build_generator(MLP(64, 10))
        latent = ArbitraryLatent(
            (1, 64), init_fn=partial(nn.init.normal_, mean=0, std=1)
        )
        optimizer_factory = build_optimizer_factory(optim.SGD, lr=0.1)
        scheduler_factory = build_scheduler_factory(
            optim.lr_scheduler.StepLR, step_size=1, gamma=0.1
        )
        optimizer = optimizer_factory(generator, latent)
        scheduler = scheduler_factory(optimizer)
        self.assertIsInstance(
            scheduler,
            optim.lr_scheduler.StepLR,
            msg="Scheduler factory should return an instance of optim.lr_scheduler.LRScheduler",
        )

        latent.reset_states()
        generator.reset_states()
        optimizer.zero_grad()
        output = generator(latent())
        loss = output.sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        self.assertEqual(
            optimizer.param_groups[0]["lr"],
            0.1 * 0.1,
            "Scheduler does not update the learning rate correctly.",
        )

        # check if reference to the optimizer is kept during re-initialization
        for _ in range(10):
            optimizer = optimizer_factory(generator, latent)
            scheduler = scheduler_factory(optimizer)
        else:
            self.assertTrue(
                scheduler.optimizer is optimizer,
                "Scheduler should keep the reference to the optimizer during re-initialization.",
            )


if __name__ == "__main__":
    unittest.main()
