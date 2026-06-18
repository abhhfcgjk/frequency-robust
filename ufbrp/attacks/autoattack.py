import math
from collections.abc import Callable

from . import  checks
import numpy as np
import torch
from .autopgd import APGDAttack, APGDAttack_targeted
from .base import Attacker
from .fab_pt import FABAttack_PT
from .square import SquareAttack
from .state import EvaluationState
from torch import Tensor


class AutoAttack(Attacker):
    def __init__(
        self, model, loss_computer: Callable[..., Tensor], eps=3.0, attacks_to_run=[], device="cuda", **kwargs
    ):
        super().__init__(model, device)
        self.loss = loss_computer
        self.norm = "Linf"
        self.epsilon = eps / 255
        self.attacks_to_run = attacks_to_run
        self.bs = 250

        self.apgd = APGDAttack(
            self.model, self.loss, n_restarts=5, n_iter=100, eps=self.epsilon, eot_iter=1, rho=0.75, device=self.device
        )

        self.fab = FABAttack_PT(
            self.model,
            n_restarts=5,
            n_iter=100,
            eps=self.epsilon,
            seed=20,
            norm=self.norm,
            verbose=False,
            device=self.device,
        )

        self.square = SquareAttack(
            self.model,
            p_init=0.8,
            n_queries=5000,
            eps=self.epsilon,
            norm=self.norm,
            n_restarts=1,
            verbose=False,
            device=self.device,
            resc_schedule=False,
        )

        self.apgd_targeted = APGDAttack_targeted(
            self.model, n_restarts=1, n_iter=100, eps=self.epsilon, eot_iter=1, rho=0.75, device=self.device
        )

        self.attacks_to_run = ["apgd-ce", "apgd-t", "fab-t", "square"]
        self.apgd.n_restarts = 1
        self.apgd_targeted.n_target_classes = 9
        self.fab.n_restarts = 1
        self.apgd_targeted.n_restarts = 1
        self.fab.n_target_classes = 9
        self.square.n_queries = 5000

    def get_logits(self, x):
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        #     logits = self.model(x)
        # return logits.float()   # keep attack math in fp32
        return self.model(x)

    def run(self, x_orig, y_orig):
        state = EvaluationState(set(self.attacks_to_run), path=None)
        state.to_disk()
        bs = self.bs

        attacks_to_run = list(filter(lambda attack: attack not in state.run_attacks, self.attacks_to_run))

        checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device), y_orig[:bs].to(self.device), bs=bs)
        n_cls = checks.check_range_output(
            self.get_logits,
            x_orig[:bs].to(self.device),
        )
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device))
        checks.check_n_classes(
            n_cls,
            self.attacks_to_run,
            self.apgd_targeted.n_target_classes,
            self.fab.n_target_classes,
        )

        with torch.no_grad():
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])

                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx:end_idx] = output
                    correct_batch = y.eq(output)
                    robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": robust_accuracy}
                state.clean_accuracy = robust_accuracy

            else:
                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": state.clean_accuracy}

            x_adv = x_orig.clone().detach()
            for attack in attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)

                    # run attack
                    if attack == "apgd-ce":
                        adv_curr = self.apgd.run(x, y)
                    elif attack == "square":
                        # square
                        adv_curr = self.square.perturb(x, y)
                    elif attack == "apgd-t":
                        # targeted apgd
                        adv_curr = self.apgd_targeted.run(x, y)
                    elif attack == "fab-t":
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        adv_curr = self.fab.perturb(x, y)
                    else:
                        raise ValueError("Attack not supported")

                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    state.robust_flags = robust_flags

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)

            # check about square
            checks.check_square_sr(robust_accuracy_dict)
            state.to_disk(force=True)
        return x_adv

    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.0
        for counter in range(n_batches):
            x = x_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        return acc.item() / x_orig.shape[0]
