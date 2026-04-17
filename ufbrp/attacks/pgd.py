from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks.base import Attacker
from torch import Tensor


def norm_loss_with_normalization(
    y_pred: torch.tensor,
    y: torch.tensor,
    alpha: list[float] = [1, 0],
    p: float = 1.0,
    q: float = 2.0,
    detach: bool = False,
    exponent: bool = True,
) -> torch.tensor:
    """norm_loss_with_normalization: norm-in-norm"""
    batch_size = y_pred.size(0)
    if batch_size > 1:
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)
        y_pred = y_pred / (1e-8 + normalization)
        y = y - torch.mean(y)
        y = y / (1e-8 + torch.norm(y, p=q))
        scale = np.power(2, max(1, 1.0 / q)) * np.power(batch_size, max(0, 1.0 / p - 1.0 / q))  # p, q>0
        loss0, loss1 = 0, 0

        if alpha[0] > 0:
            err = y_pred - y
            # avoid gradient explosion when 0 <= p <1;
            # and avoid vanishing gradient problem when p < 0
            if p < 1:
                err += 1e-8
            # loss0 = torch.norm(err, p=p) / scale  # Actually, p = q = 2 is related to PLCC
            loss0 = torch.pow(err, p) / scale
            loss0 *= torch.sign(err)
            # loss0 = torch.pow(loss0, p) if exponent else loss0  #
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.t(), y.t())
            err = rho * y_pred - y
            # avoid gradient explosion when 0 <= p <1;
            # and avoid vanishing gradient problem when p < 0
            if p < 1:
                err += 1e-8
            loss1 = torch.norm(err, p=p) / scale  # Actually, p = q = 2 is related to LSR
            loss1 = torch.pow(loss1, p) if exponent else loss1  #  #
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


class PGD(Attacker):
    """
    Fast Gradient Sign Attack.

    :param model: The model to be attacked. Must be on device (.cuda()).
    :param scaler: The loss scaler.
    :param eps: Upper bound for linf norm of perturbations.
    :param alpha: Step size which is multiplied by the gradient sign. It is
        reasonable to choose values slightly greater than eps as shown by fast
        training paper.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_computer: Callable[..., Tensor],
        eps: float,
        alpha: float = None,
        iters: int = 10,
        mode: str = "zero",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model)
        self.eps = eps / 255
        self.alpha = alpha / 255 if alpha is not None else self.eps
        self.iters = iters

        self.loss_computer = loss_computer
        self.mode = mode
        self.noise: Tensor | None = None

    def run(self, inputs: Tensor, target: Tensor):
        noise = torch.empty(*inputs.shape, device=inputs.device)
        noise.uniform_(-self.eps, self.eps)

        for _ in range(self.iters):
            noise_with_grad = noise.detach().requires_grad_()
            noisy_inputs = inputs + noise_with_grad
            noisy_inputs.clamp_(0.0, 1.0)

            noisy_outputs = self.model(noisy_inputs)
            loss = self.loss_computer(noisy_outputs, target)

            grad = torch.autograd.grad(loss, [noise_with_grad])[0].detach()

            noise += self.alpha * torch.sign(grad)
            noise.clamp_(-self.eps, self.eps)

        attacked = inputs + noise
        attacked.clamp_(0.0, 1.0)
        return attacked


class AutoPGD(Attacker):
    """Auto-PGD: A budget-aware step size-free variant of PGD.

    :param model: The model to be attacked. Must be on device (.cuda()).
    :param k: Number of PGD iterations per single initialization (restart).
    :param eps: Upper bound for linf norm of perturbations.
    :param alpha: Step size. In PGD, it is commonly much lower than eps.
    :param restarts: Number of algorithm restarts, i.e. the number of different
        random initialization points.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_computer: Callable[..., Tensor],
        iters: int = 2,
        eps: float = 4.0,
        alpha: float = 1.0,
        thr_decr: float = 0.75,
        *args,
        **kwargs,
    ):
        super().__init__(model)

        self.n_iter = iters
        self.n_iter_min = max(int(0.06 * iters), 1)
        self.size_decr = max(int(0.03 * iters), 1)
        self.k = max(int(0.22 * iters), 1)

        self.eps = eps / 255
        self.alpha = alpha
        self.thr_decr = thr_decr
        self.loss_computer = loss_computer

    def check_oscillation(self, loss_steps, cur_step):
        t = torch.zeros(loss_steps.shape[1], device=loss_steps.device, dtype=loss_steps.dtype)
        for i in range(self.k):
            t += (loss_steps[cur_step - i] > loss_steps[cur_step - i - 1]).float()

        return (t <= self.k * self.thr_decr * torch.ones_like(t)).float()

    def run(self, inputs: Tensor, target: Tensor):
        self.k = max(int(0.22 * self.n_iter), 1)
        device = inputs.device

        x_adv = inputs.clone()
        x_best = x_adv.clone().detach()
        loss_steps = torch.zeros([self.n_iter, inputs.shape[0]], device=device)
        loss_best_steps = torch.zeros([self.n_iter + 1, inputs.shape[0]], device=device)

        step_size = (
            self.alpha
            * self.eps
            * torch.ones([inputs.shape[0], *[1] * (len(inputs.shape) - 1)], device=device, dtype=inputs.dtype)
        )

        counter3 = 0

        # 1 step of the classic PGD
        x_adv.requires_grad_()

        logits = self.model(x_adv)
        loss_indiv = self.loss_computer(logits, target)
        loss = loss_indiv.sum()

        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        grad_best = grad.clone()
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()

        loss_best = loss_indiv.detach().clone()
        loss_best = loss_best.squeeze(-1)
        loss_best_last_check = loss_best.clone()
        loss_best_last_check = loss_best_last_check.squeeze(-1)
        reduced_last_check = torch.ones_like(loss_best)

        x_adv_old = x_adv.clone().detach()

        for i in range(self.n_iter):
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()

            a = 0.75 if i > 0 else 1.0

            x_adv_1 = x_adv + step_size * torch.sign(grad)
            x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, inputs - self.eps), inputs + self.eps), 0.0, 1.0)
            x_adv_1 = torch.clamp(
                torch.min(
                    torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), inputs - self.eps),
                    inputs + self.eps,
                ),
                0.0,
                1.0,
            )

            x_adv = x_adv_1

            if i < self.n_iter - 1:
                x_adv.requires_grad_()

            logits = self.model(x_adv)
            loss_indiv = self.loss_computer(logits, target)

            loss = loss_indiv.sum()

            if i < self.n_iter - 1:
                grad = torch.autograd.grad(loss, [x_adv])[0].detach()

            x_adv.detach_()
            loss_indiv.detach_()
            loss.detach_()

            # check step size
            y1 = loss_indiv.detach().clone()
            y1 = y1.squeeze(-1)
            loss_steps[i] = y1
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind]
            loss_best_steps[i + 1] = loss_best

            counter3 += 1

            if counter3 == self.k:
                fl_oscillation = self.check_oscillation(loss_steps, i)
                fl_reduce_no_impr = (1.0 - reduced_last_check) * (loss_best_last_check >= loss_best).float()
                fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                reduced_last_check = fl_oscillation.clone()
                loss_best_last_check = loss_best.clone()

                if fl_oscillation.sum() > 0:
                    ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                    step_size[ind_fl_osc] /= 2.0

                    x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                    grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                counter3 = 0
                self.k = max(self.k - self.size_decr, self.n_iter_min)

        # print(self.k)
        return x_best
