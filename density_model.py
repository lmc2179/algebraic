import math

import numpy as np
from scipy.special import comb
import von_mises_fisher
from algebraic_statistic import Mean, Variance, AbstractCompositeGroupStatistic

class AbstractDensityModel(AbstractCompositeGroupStatistic):
    def pdf(self, *args):
        raise NotImplementedError

    def log_pdf(self, *args):
        raise NotImplementedError


class NormalDistribution(AbstractDensityModel):
    STATISTIC_CLASSES = [('mean', Mean), ('variance', Variance)]

    def pdf(self, X):
        return self._calculate_normalizing_constant() * math.exp(self._mahalanobis_distance(X))

    def _calculate_normalizing_constant(self):
        var = self['variance'].get_variance()
        return 1.0 / (math.sqrt(2 * math.pi * var))

    def _mahalanobis_distance(self, X):
        mu = self['mean'].get_mean()
        var = self['variance'].get_variance()
        return - ((X - mu)**2) / (2 * var)

    def unnormalized_pdf(self, X):
        return self._mahalanobis_distance(X)


class PoissonDistribution(AbstractDensityModel):
    STATISTIC_CLASSES = [('mean', Mean)]

    def pdf(self, k):
        lam = self['mean'].get_mean()
        return (lam**k * math.exp(-lam)) / (math.factorial(k))

    def unnormalized_pdf(self, k):
        lam = self['mean'].get_mean()
        return lam**k


class BernoulliDistribution(AbstractDensityModel):
    STATISTIC_CLASSES = [('mean', Mean)]

    def pdf(self, x):
        mu = self['mean'].get_mean()
        return (mu ** x)*((1.0 - mu) ** (1 - x))

    def unnormalized_pdf(self, x):
        return self.pdf(x)


class BinomialDistribution(AbstractDensityModel):
    STATISTIC_CLASSES = [('mean', Mean)]

    def pdf(self, n, k):
        # n is the number of trials, k is the number of successes
        mu = self['mean'].get_mean()
        return comb(n, k) * (mu ** k) * ((1.0- mu) ** (n - k))

    def unnormalized_pdf(self, n, k):
        return self.pdf(n, k)


class ExponentialDistribution(AbstractDensityModel):
    STATISTIC_CLASSES = [('mean', Mean)]

    def pdf(self, x):
        lam = 1.0 / self['mean'].get_mean()
        return lam * math.exp(-lam * x)

    def unnormalized_pdf(self, x):
        return self.pdf(x)


# class vonMisesFisherDistribution(AbstractDensityModel):
#     STATISTIC_CLASSES = [('mean', Mean)]
#
#     def pdf(self, x):
#         lam = 1.0 / self['mean'].get_mean()
#         return lam * math.exp(-lam * x)
#
#     def _calculate_kappa(self, ):
#
#     def unnormalized_pdf(self, x):
#         return self.pdf(x)

class CategoricalDistribution(AbstractDensityModel):
    STATISTIC_CLASSES = [('mean', Mean)]

    def pdf(self, x):
        mean = self['mean'].get_mean()
        return np.dot(x, mean)

    def unnormalized_pdf(self, x):
        # Technically this one is normalized
        mean = self['mean'].get_mean()
        return np.dot(x, mean)

class BetaDistribution(AbstractDensityModel):
    pass