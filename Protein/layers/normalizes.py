import torch


def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
    # create distribution Normal
    normal = torch.distributions.normal.Normal(0, 1)
    # 将截断区间 [a, b] 标准化为标准正态分布下的 alpha 和 beta
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    # 计算 alpha 和 beta 在标准正态分布下的累积分布函数值（CDF）
    alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
    beta_normal_cdf = normal.cdf(torch.tensor(beta))
    # 将 uniform 从 [0,1] 映射到 [CDF(alpha), CDF(beta)] 区间
    p = alpha_normal_cdf + (beta_normal_cdf - alpha_normal_cdf) * uniform
    v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
    x = mu + sigma * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(v)
    x = torch.clamp(x, a, b)
    return x


def sample_truncated_normal(shape):
    stddev = torch.sqrt(torch.tensor(1.0 / shape[-1])) / 0.87962566103423978
    return stddev * truncated_normal(torch.rand(shape))


def init_lecun_normal(module):
    module.weight = torch.nn.Parameter((sample_truncated_normal(module.weight.shape)))
    return module
