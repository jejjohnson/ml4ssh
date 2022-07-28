"""
Taken from:
* https://github.com/didriknielsen/survae_flows/blob/master/survae/distributions/base.py
* https://github.com/didriknielsen/survae_flows/blob/master/survae/distributions/conditional/base.py
* https://github.com/didriknielsen/survae_flows/blob/master/survae/distributions/conditional/normal.py
"""

import math
import torch
from torch import nn
from torch.distributions import Normal


class ConditionalDistribution(nn.Module):
    """Distribution base class."""

    def log_prob(self, x, context):
        """Calculate log probability under the distribution.
        Args:
            x: Tensor, shape (batch_size, ...).
            context: Tensor, shape (batch_size, ...).
        Returns:
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def sample(self, context):
        """Generates samples from the distribution.
        Args:
            context: Tensor, shape (batch_size, ...).
        Returns:
            samples: Tensor, shape (batch_size, ...).
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, context):
        """Generates samples from the distribution together with their log probability.
        Args:
            context: Tensor, shape (batch_size, ...).
        Returns::
            samples: Tensor, shape (batch_size, ...).
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def forward(self, *args, mode, **kwargs):
        """
        To allow Distribution objects to be wrapped by DataParallelDistribution,
        which parallelizes .forward() of replicas on subsets of data.
        DataParallelDistribution.log_prob() calls DataParallel.forward().
        DataParallel.forward() calls Distribution.forward() for different
        data subsets on each device and returns the combined outputs.
        """
        if mode == "log_prob":
            return self.log_prob(*args, **kwargs)
        else:
            raise RuntimeError("Mode {} not supported.".format(mode))


class ConditionalMeanNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and fixed std."""

    def __init__(self, net, scale=1.0):
        super(ConditionalMeanNormal, self).__init__()
        self.net = net
        self.scale = scale

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.scale)

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return dist.log_prob(x)

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)

        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalMeanStdNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and learned std."""

    def __init__(self, net, scale_shape):
        super(ConditionalMeanStdNormal, self).__init__()
        self.net = net
        self.log_scale = nn.Parameter(torch.zeros(scale_shape))

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.log_scale.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return dist.log_prob(x)

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1):
        super(ConditionalNormal, self).__init__()
        self.net = net
        self.split_dim = split_dim

    def cond_dist(self, context):
        params = self.net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        return Normal(loc=mean, scale=log_std.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return dist.log_prob(x)

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev


def sum_except_batch(x, num_dims=1):
    """
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    """
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)
