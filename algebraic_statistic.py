"""
This module shows how the sample average can be formulated as an algebraic statistic, with a
"merge" operation ("|") and an inverse.
"""

import abc
import copy

class AbstractProbabilityDistribution(object):
    """
    This class describes the API for a probability distribution.

    An estimate for a probability distribution is assumed to be constructed of one or more statistics.
    The specification of these statistics defines the algebraic behavior of the distribution totally,
    and subclasses should only need to implement that calculation of the probability measure from the statistics.

    The use of composition here is designed to separate the business of calculating the statistic values
    from that of calculating the probabilistic qualities of the distribution.
    """
    def __init__(self, data):
        self.statistics = self._construct_statistics(data)

    def _construct_statistics(self, data):
        """
        Subclasses should implement this method, returning a dictionary mapping statistic names to statistic objects.
        """
        raise NotImplementedError

    def pdf(self, X):
        """The PDF (or PMF) at the point X."""
        raise NotImplementedError

    def log_pdf(self, X):
        """The log PDF (or PMF) at the point X."""
        raise NotImplementedError

    def __or__(self, other):
        pass # TODO: Merge all statistics

    def __neg__(self):
        pass # TODO: Er...which interface is this implementing?

class AbstractGroupStatistic(object):
    def __init__(self, data=None):
        """
        The statistic should be calculated and saved to the object's state here.
        """
        raise NotImplementedError

    def __or__(self, other):
        """
        This is the binary operation combining the two statistics.
        """
        raise NotImplementedError

    @classmethod
    def get_identity(cls):
        "The identity element should be produced when __init__ is called with None as an argument."
        return cls()

    def is_identity(self):
        raise NotImplementedError

    def __neg__(self):
        """
        Returns the inverse version of this element.
        """
        raise NotImplementedError

    def __sub__(self, other):
        return self | -other

class Mean(AbstractGroupStatistic):
    def __init__(self, data=None):
        if data is None:
            self.n = 0
            self.mean = 0
        else:
            self.n = len(data)
            self.mean = 1.0*sum(data) / len(data)

    def get_mean(self):
        return self.mean

    def get_n(self):
        return self.n

    def set_mean(self, new_mean):
        self.mean = new_mean

    def set_n(self, new_n):
        self.n = new_n

    def __or__(self, other):
        if self == -other: # This handles the case where the element is merged with its inverse, but it's really silly
            return Mean()
        merged_m = Mean()
        merged_n = self.get_n() + other.get_n()
        merged_mean_val = 1.0/merged_n * (self.get_n() * self.get_mean() + other.get_n() * other.get_mean())
        merged_m.set_n(merged_n)
        merged_m.set_mean(merged_mean_val)
        return merged_m

    def __eq__(self, other):
        return self.n == other.n and self.mean == other.mean

    def __neg__(self):
        new_mean = Mean()
        new_mean.set_mean(self.get_mean())
        new_mean.set_n(-self.get_n())
        return new_mean

class Variance(AbstractGroupStatistic):
    def __init__(self, data=None):
        self.mean = Mean(data)
        if data is None:
            self.sum_square_distance = 0
        else:
            self.sum_square_distance = sum([(d - self.mean.get_mean())**2 for d in data])

    def get_sum_square_distance(self):
        return self.sum_square_distance

    def get_variance(self):
        return 1.0/(self.mean.get_n()-1) * self.sum_square_distance

    def __or__(self, other):
        v = Variance()
        v.mean = self.mean | other.mean
        v.sum_square_distance =   self.get_sum_square_distance() + self.mean.get_n() * self.mean.get_mean()**2 \
                                + other.get_sum_square_distance() + other.mean.get_n() * other.mean.get_mean()**2 \
                                - v.mean.get_n() * v.mean.get_mean()**2
        return v

    def __eq__(self, other):
        return self.get_sum_square_distance() == other.get_sum_square_distance()

    def __neg__(self):
        new_variance = copy.deepcopy(self)
        new_variance.sum_square_distance *= -1
        new_variance.mean.n *= -1
        return new_variance