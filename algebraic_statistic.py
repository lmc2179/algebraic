from collections import Counter
import copy

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
        merged_m = Mean()
        merged_n = self.get_n() + other.get_n()
        if merged_n == 0: # Handle subtracting all elements
            return Mean()
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

class Frequency(AbstractGroupStatistic):
    """
    Keeps track of the count of each object observed in the data so far. You can think of this as the
    collections.Counter class endowed with the algebraic properties of a group.
    """
    def __init__(self, data=None):
        if data is None:
            self.counter = Counter()
        else:
            self.counter = Counter(data)

    def get_counter(self):
        return self.counter

    def set_counter(self, new_counter):
        self.counter = new_counter

    def get_frequency(self, obj):
        if obj in self.counter:
            return self.counter[obj]
        else:
            return 0

    def __or__(self, other):
        self_ctr = self.get_counter()
        other_ctr = other.get_counter()
        unique_tokens = list(set(self_ctr.keys()) | set(other_ctr.keys()))
        merged_counter_sums = ((k, self.get_frequency(k) + other.get_frequency(k)) for k in unique_tokens)
        merged_counter = dict((k,v) for k,v in merged_counter_sums if v > 0)
        result = Frequency()
        result.set_counter(merged_counter)
        return result

    def __neg__(self):
        neg_ctr = copy.deepcopy(self.get_counter())
        for k in neg_ctr.keys():
            neg_ctr[k] = -neg_ctr[k]
        result = Frequency()
        result.set_counter(neg_ctr)
        return result

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

class AbstractCompositeGroupStatistic(AbstractGroupStatistic):
    STATISTIC_CLASSES = None
    def __init__(self, data=None):
        self.statistic_values = {name: cls(data) for name, cls in self.STATISTIC_CLASSES}

    def __or__(self, other):
        result = self.__class__()
        names = self.statistic_values.keys()
        statistic_values = {n: self.statistic_values[n] | other.statistic_values[n] for n in names}
        result.statistic_values = statistic_values
        return result

    def __neg__(self):
        result = self.__class__()
        names = self.statistic_values.keys()
        statistic_values = {n: -self.statistic_values[n] for n in names}
        result.statistic_values = statistic_values
        return result

    def __eq__(self, other):
        names = self.statistic_values.keys()
        result = all([self.statistic_values[n] == other.statistic_values[n] for n in names])
        return result

    def __getitem__(self, key):
        return self.statistic_values[key]