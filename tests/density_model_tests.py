import math
import random

import numpy as np

from density_model import NormalDistribution, PoissonDistribution, CategoricalDistribution, BernoulliDistribution, \
    ExponentialDistribution, BinomialDistribution
from tests.algebraic_statistic_tests import AbstractGroupStatisticTest


class NormalDistributionTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = NormalDistribution
    def _generate_data_set(self, size):
        return [random.randint(-1000, 1000) for i in range(size)]

    def _assert_equal(self, s1, s2):
        self.assertAlmostEqual(s1.statistic_values['variance'].get_variance(),
                               s2.statistic_values['variance'].get_variance(), places=6)
        self.assertAlmostEqual(s1.statistic_values['mean'].get_mean(),
                               s2.statistic_values['mean'].get_mean(), places=6)

    def test_pdf(self):
        TRUE_MU = 0
        TRUE_VAR = 1
        D = np.random.normal(TRUE_MU, TRUE_VAR, 50000)
        n = NormalDistribution(D)
        true_pdf = lambda x : 1.0 / (math.sqrt(2 * math.pi * TRUE_VAR)) * math.exp(- ((x - TRUE_MU)**2) / (2 * TRUE_VAR))
        for i in range(-3,3):
            self.assertAlmostEqual(true_pdf(i), n.pdf(i), places=2)


class PoissonDistributionTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = PoissonDistribution
    def _generate_data_set(self, size):
        return [random.randint(0, 1000) for i in range(size)]

    def _assert_equal(self, s1, s2):
        self.assertAlmostEqual(s1.statistic_values['mean'].get_mean(),
                               s2.statistic_values['mean'].get_mean(), places=6)

    def test_pdf(self):
        TRUE_LAMBDA = 1.2
        D = np.random.poisson(lam=TRUE_LAMBDA, size=5000)
        p = PoissonDistribution(D)
        true_pmf = lambda k : (TRUE_LAMBDA**k * math.exp(-TRUE_LAMBDA)) / (math.factorial(k))
        for i in range(0,10):
            self.assertAlmostEqual(true_pmf(i), p.pdf(i), places=2)


class CategoricalDistributionTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = CategoricalDistribution
    TEST_DATA_ITEM_LENGTH = 10
    def _generate_data_set(self, size):
        int_data = [random.randint(0, self.TEST_DATA_ITEM_LENGTH) for i in range(size)]
        encoded_data = [self._label_encode(x, 10) for x in int_data]
        return encoded_data

    def _label_encode(self, x, max_val):
        return np.array([1 if i == x else 0 for i in range(max_val)])

    def _assert_equal(self, s1, s2):
        m1, m2 = s1['mean'], s2['mean']
        self.assertTrue(self._almost_equal(m1.get_mean(), m2.get_mean()))
        self.assertEqual(m1.get_n(), m2.get_n())

    def _almost_equal(self, v1, v2):
        return np.all(np.isclose(v1, v2))

    def test_pdf(self):
        # PDF is uniform here
        TRUE_CATEGORY_LIKELIHOOD = 1.0/ self.TEST_DATA_ITEM_LENGTH
        D = self._generate_data_set(50000)
        p = CategoricalDistribution(D)
        for i in range(0,self.TEST_DATA_ITEM_LENGTH):
            test_x = [1 if i == j else 0 for j in range(self.TEST_DATA_ITEM_LENGTH)]
            self.assertAlmostEqual(p.pdf(test_x), TRUE_CATEGORY_LIKELIHOOD, places=1)


class BernoulliDistributionTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = BernoulliDistribution
    TEST_DATA_MU = 0.6
    def _generate_data_set(self, size):
        return [1 if random.random() < self.TEST_DATA_MU else 0 for i in range(size)]

    def _assert_equal(self, s1, s2):
        m1, m2 = s1['mean'], s2['mean']
        self.assertAlmostEqual(m1.get_mean(), m2.get_mean())

    def test_pdf(self):
        # PDF is uniform here
        D = self._generate_data_set(50000)
        p = BernoulliDistribution(D)
        self.assertAlmostEqual(p.pdf(0), 1.0 - self.TEST_DATA_MU, places=2)
        self.assertAlmostEqual(p.pdf(1), self.TEST_DATA_MU, places=2)


class ExponentialDistributionTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = ExponentialDistribution
    def _generate_data_set(self, size):
        return [random.randint(0, 1000) for i in range(size)]

    def _assert_equal(self, s1, s2):
        self.assertAlmostEqual(s1.statistic_values['mean'].get_mean(),
                               s2.statistic_values['mean'].get_mean(), places=6)

    def test_pdf(self):
        TRUE_LAMBDA = 1.2
        D = np.random.exponential(scale=1.0/TRUE_LAMBDA, size=50000)
        p = ExponentialDistribution(D)
        true_pmf = lambda x : TRUE_LAMBDA * math.exp(-TRUE_LAMBDA * x)
        for i in range(0,10):
            self.assertAlmostEqual(true_pmf(i), p.pdf(i), places=2)


class BinomialDistributionTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = BernoulliDistribution
    TEST_DATA_MU = 0.6
    def _generate_data_set(self, size):
        return [1 if random.random() < self.TEST_DATA_MU else 0 for i in range(size)]

    def _assert_equal(self, s1, s2):
        m1, m2 = s1['mean'], s2['mean']
        self.assertAlmostEqual(m1.get_mean(), m2.get_mean())

    def test_pdf(self):
        # PDF is uniform here
        D = self._generate_data_set(50000)
        p = BinomialDistribution(D)
        self.assertAlmostEqual(p.pdf(1, 0), 1.0 - self.TEST_DATA_MU, places=2)
        self.assertAlmostEqual(p.pdf(1, 1), self.TEST_DATA_MU, places=2)

#
# class vonMisesFisherDistributionTest(AbstractGroupStatisticTest):
#     STATISTIC_CLS = vonMisesFisherDistribution
#     def _generate_data_set(self, size):
#         return [np.linalg.norm(np.array([random.random(), random.random()])) for i in range(size)]
#
#     def _assert_equal(self, s1, s2):
#         m1, m2 = s1['mean'], s2['mean']
#         self.assertAlmostEqual(m1.get_mean(), m2.get_mean())
#
# if __name__ == '__main__':
#     unittest.main()