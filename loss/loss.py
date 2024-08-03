import torch
import torch.nn.functional as F
import numpy as np
def compute_mse(labels_1hot_, evi_alp_):
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)

    loss_mse_ = (labels_1hot_ - evi_alp_ / evi_alp0_).pow(2).sum(-1).mean()
    loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(
        -1).mean()

    return loss_mse_, loss_var_

def compute_fisher_mse(labels_1hot_, evi_alp_):
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)
    # torch.polygamma 计算一阶的多项式 Gamma 函数。这函数通常用于统计学或数学中的一些特定问题，例如处理超趋势（super-trend）指标等。
    gamma1_alp = torch.polygamma(1, evi_alp_)
    gamma1_alp0 = torch.polygamma(1, evi_alp0_)

    gap = labels_1hot_ - evi_alp_ / evi_alp0_

    loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1).mean()

    loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum \
        (-1).mean()

    loss_det_fisher_ = - (torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))).mean()

    return loss_mse_, loss_var_, loss_det_fisher_

def compute_kl_loss(alphas, labels, target_concentration, concentration=1.0, epsilon=1e-8):
    # TODO: Need to make sure this actually works right...
    # todo: so that concentration is either fixed, or on a per-example setup

    # Create array of target (desired) concentration parameters
    if target_concentration < 1.0:
        concentration = target_concentration

    target_alphas = torch.ones_like(alphas) * concentration
    target_alphas += torch.zeros_like(alphas).scatter_(-1, labels.unsqueeze(-1), target_concentration - 1)

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                          torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = torch.squeeze(alp0_term + alphas_term).mean()

    return loss

def loss(labels_,evi_alp_,flag,kl_c=-1,target_con=1,epoch=1,fisher_c=1):
    labels_1hot_ = torch.eye(60)[labels_]
    if flag == 'IEDL':
        # IEDL -> fisher_mse
        loss_mse_, loss_var_, loss_fisher_ = compute_fisher_mse(labels_1hot_, evi_alp_)

    elif flag == 'EDL':
        # EDL -> mse
        loss_mse_, loss_var_ = compute_mse(labels_1hot_, evi_alp_)

    elif flag == 'DEDL':
        loss_mse_, loss_var_ = compute_mse(labels_1hot_, evi_alp_)
        _, _, loss_fisher_ = compute_fisher_mse(labels_1hot_, evi_alp_)

    else:
        raise NotImplementedError

    evi_alp_ = (evi_alp_ - target_con) * (1 - labels_1hot_) + target_con
    loss_kl_ = compute_kl_loss(evi_alp_, labels_, target_con)

    if kl_c == -1:
        regr = np.minimum(1.0, epoch / 10.)
        grad_loss = loss_mse_ + loss_var_ + fisher_c * loss_fisher_ + regr * loss_kl_
    else:
        grad_loss = loss_mse_ + loss_var_ + fisher_c * loss_fisher_ + kl_c * loss_kl_

    return grad_loss
if __name__ == '__main__':
    x = torch.randn(10, 60)
    x = F.relu(x)
    y = torch.randint(60, size=(10,))
    loss(y,x+1,'IEDL')