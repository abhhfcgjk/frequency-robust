import torch
from .base import Attacker


def check_zero_gradients(grad, logger=None):
    _ = grad.view(grad.shape[0], -1).abs().sum(-1)


def L1_projection(x2, y2, eps1):
    """
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    """

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    # u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2 * (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)

    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)

    if c2.nelement != 0:

        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)

        # print(c2.shape, lb.shape)

        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.0)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            # print(ind3.shape)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            # print(lb, ub)
            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)


class APGDAttack(Attacker):
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param model:       forward pass function
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
        self, model, loss_computer, n_iter=100, n_restarts=1, eps=3.0, eot_iter=1, rho=0.75, topk=None, device="cuda"
    ):
        """
        AutoPGD implementation in PyTorch
        """
        super().__init__(model, device)
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.n_restarts = n_restarts
        if loss_computer is None:
            self.loss = self.dlr_loss_targeted
        else:
            self.loss = loss_computer
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.device = device
        self.use_rs = True
        self.n_iter_orig = n_iter
        self.eps_orig = eps / 255
        self.y_target = None

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def init_hyperparam(self, x):

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        t = x.abs().view(x.shape[0], -1).max(1)[0]
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1.0 - ind)) / (
            x_sorted[:, -1] - x_sorted[:, -3] + 1e-12
        )

    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
        x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t)

        if x_init is not None:
            x_adv = x_init.clone()

        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss_indiv = self.loss(logits, y)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        alpha = 2.0
        step_size = alpha * self.eps * torch.ones([x.shape[0], *([1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        k = self.n_iter_2 + 0
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                a = 0.75 if i > 0 else 1.0

                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(
                    torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps),
                    0.0,
                    1.0,
                )
                x_adv = x_adv_1 + 0.0

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = self.loss(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.0
            ### check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0
                counter3 += 1
                if counter3 == k:
                    fl_oscillation = self.check_oscillation(loss_steps, i, k, loss_best, k3=self.thr_decr)
                    fl_reduce_no_impr = (1.0 - reduced_last_check) * (loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()
                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()
                    k = max(k - self.size_decr, self.n_iter_min)
                    counter3 = 0
        return (x_best, acc, loss_best, x_best_adv)

    def dlr_loss_targeted(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        u = torch.arange(x.shape[0])
        return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - 0.5 * (x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

    def run(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        if y is not None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        if y is None:
            # y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        acc = y_pred == y
        if not best_loss:
            for _ in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float("inf"))
            for _ in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.0
                loss_best[ind_curr] = loss_curr[ind_curr] + 0
            return adv_best


class APGDAttack_targeted(APGDAttack):
    def __init__(
        self,
        model,
        n_iter=100,
        n_restarts=1,
        eps=None,
        eot_iter=1,
        rho=0.75,
        topk=None,
        n_target_classes=9,
        device=None,
    ):
        """
        AutoPGD on the targeted DLR loss
        """
        super().__init__(
            model,
            loss_computer=None,
            n_iter=n_iter,
            n_restarts=n_restarts,
            eps=eps,
            eot_iter=eot_iter,
            rho=rho,
            topk=topk,
            device=device,
        )
        self.y_target = None
        self.n_target_classes = n_target_classes

    def dlr_loss_targeted(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        u = torch.arange(x.shape[0])
        return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - 0.5 * (x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

    def run(self, x, y=None, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        """
        if y is not None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        if y is None:
            # y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        acc = y_pred == y

        for target_class in range(2, self.n_target_classes + 2):
            for _ in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    output = self.model(x_to_fool)
                    self.y_target = output.sort(dim=1)[1][:, -target_class]

                    res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
        return adv
