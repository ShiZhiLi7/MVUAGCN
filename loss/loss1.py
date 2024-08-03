import torch
Device = "cuda:0"
def KL(alpha):
    '''
    这段代码计算了两个多项分布参数 alpha 和 beta 之间的KL散度（Kullback-Leibler divergence）。KL散度用于衡量两个概率分布之间的相似性或差异性，对于多项分布，它衡量了一个多项分布相对于另一个多项分布的不确定性。
    KL（Kullback-Leibler）散度，也称为相对熵，是一种用于测量两个概率分布之间差异的方法。KL散度衡量了一个概率分布相对于另一个概率分布的不确定性。在信息论和统计学中，KL散度通常用于比较两个概率分布之间的相似性或差异性。
    '''
    k = alpha.size(1)  # 获取多项分布的维度

    beta = torch.ones((1, k), dtype=torch.float32).to(Device)

    # 计算 alpha 和 beta 中每个多项分布的总和
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    # 计算对数的贝塔函数和对数的均匀多项分布的贝塔函数
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    # 计算KL散度
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

    return kl

def mse_loss(p, alpha, global_step,annealing_step=10):
    n_class = alpha.size(1)
    label = torch.eye(n_class).to(Device)[p]
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp)

    loss = (A + B) + C
    loss = torch.mean(loss)
    return loss

def ce_loss(alpha,p, global_step,annealing_step=10,):
    n_class = alpha.size(1)
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = torch.eye(n_class).to(Device)[p]
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp)
    loss = torch.mean((A + B))
    return loss


def fisher_mse(p, alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)
    label = torch.eye(60)[p]
    # torch.polygamma 计算一阶的多项式 Gamma 函数。这函数通常用于统计学或数学中的一些特定问题，例如处理超趋势（super-trend）指标等。
    gamma1_alp = torch.polygamma(1, alpha)
    gamma1_alp0 = torch.polygamma(1, S)

    gap = label - alpha / S

    loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1).mean()

    loss_var_ = (alpha * (S - alpha) * gamma1_alp / (S * S * (S + 1))).sum \
        (-1).mean()

    loss_det_fisher_ = - (torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))).mean()

    return loss_mse_, loss_var_, loss_det_fisher_

if __name__ == '__main__':
    x1 = torch.randn(10, 60)
    x2 = torch.randn(10, 60)
    x3 = torch.randn(10, 60)

    def getMax(m1, m2, m3):

        # m1 = torch.from_numpy(m1)
        # m2 = torch.from_numpy(m2)
        # m3 = torch.from_numpy(m3)

        zero_m1 = torch.zeros_like(m1)
        zero_m2 = torch.zeros_like(m2)
        zero_m3 = torch.zeros_like(m3)

        _, indices1 = torch.sort(m1, descending=True)
        _, indices2 = torch.sort(m2, descending=True)
        _, indices3 = torch.sort(m3, descending=True)
        top5_indices1 = indices1[:,0:10]
        top5_indices2 = indices2[:,0:10]
        top5_indices3 = indices3[:,0:10]
        # 找出这三个索引集合的交集
        # 注意：这里的交集可能为空或少于五个元素，取决于tensor中的实际值
        # 使用集合操作来找到交集
        for j in range(m1.size(0)):
            for i in range(5):
                if top5_indices1[j, i] in top5_indices2[j, :] and top5_indices1[j, i] in top5_indices3[j, :]:
                    zero_m1[j, top5_indices1[j, i]] = m1[j, top5_indices1[j, i]]
                if top5_indices2[j, i] in top5_indices1[j, :] and top5_indices2[j, i] in top5_indices3[j, :]:
                    zero_m2[j, top5_indices2[j, i]] = m2[j, top5_indices2[j, i]]
                if top5_indices3[j, i] in top5_indices2[j, :] and top5_indices3[j, i] in top5_indices1[j, :]:
                    zero_m3[j, top5_indices3[j, i]] = m3[j, top5_indices3[j, i]]

        return zero_m1, zero_m2, zero_m3


    getMax(x1, x2, x3)
    loss = 0
    y = torch.randint(60, size=(10,))
    loss += ce_loss(y,x+1,1)
    loss += ce_loss(y, x + 1, 1)
    loss += ce_loss(y, x + 1, 1)
    loss += ce_loss(y, x + 1, 1)
    loss1 = loss/4
    loss2 = torch.mean(loss)
    print()