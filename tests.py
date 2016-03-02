import unittest
from algebraic_statistic import Mean, Variance, NormalDistribution, PoissonDistribution, CategoricalDistribution, \
    BernoulliDistribution, ExponentialDistribution, BinomialDistribution
import random
import numpy as np
import math

class AbstractGroupStatisticTest(unittest.TestCase):
    STATISTIC_CLS = None

    def _generate_data_set(self, size):
        raise NotImplementedError

    def _generate_data_sets(self, sizes):
        return [self._generate_data_set(s) for s in sizes]

    def _partition_data_set(self, data_set, slice_sizes):
        raise NotImplementedError

    def _assert_equal(self, s1, s2):
        raise NotImplementedError

    def test_merge_correctness(self):
        d1, d2 = self._generate_data_sets([3, 4])
        merged_dataset = d1 + d2
        merged_dataset_m = self.STATISTIC_CLS(merged_dataset)
        m1, m2 = self.STATISTIC_CLS(d1), self.STATISTIC_CLS(d2)
        merged_algebraic_m = m1 | m2
        self._assert_equal(merged_algebraic_m, merged_dataset_m)

    def test_associativity(self):
        d1, d2, d3 = self._generate_data_sets([3, 4, 5])
        m1, m2, m3 = self.STATISTIC_CLS(d1), self.STATISTIC_CLS(d2), self.STATISTIC_CLS(d3)
        merged_algebraic_m_left = (m1 | m2) | m3
        merged_algebraic_m_right = m1 | (m2 | m3)
        self._assert_equal(merged_algebraic_m_left, merged_algebraic_m_right)

    def test_commutatativity(self):
        d1, d2 = self._generate_data_sets([3, 4])
        m1, m2 = self.STATISTIC_CLS(d1), self.STATISTIC_CLS(d2)
        merged_algebraic_m1_left = m1 | m2
        merged_algebraic_m1_right = m2 | m1
        self._assert_equal(merged_algebraic_m1_left, merged_algebraic_m1_right)

    def test_identity(self):
        dataset = self._generate_data_set(7)
        m = self.STATISTIC_CLS(dataset) | self.STATISTIC_CLS.get_identity()
        m_gold = self.STATISTIC_CLS(dataset)
        self._assert_equal(m, m_gold)

    def test_inverse(self):
        d1, d2 = self._generate_data_sets([3, 4])
        m1, m2 = self.STATISTIC_CLS(d1), self.STATISTIC_CLS(d2)
        merged_algebraic_m = m1 | m2 | -m2
        self._assert_equal(m1, merged_algebraic_m)

    def test_inverse_to_identity(self):
        dataset = self._generate_data_set(7)
        m = self.STATISTIC_CLS(dataset) - self.STATISTIC_CLS(dataset)
        m_ident = self.STATISTIC_CLS.get_identity()
        self._assert_equal(m, m_ident)

    def test_sub(self):
        d1, d2 = self._generate_data_sets([3, 4])
        m1, m2 = self.STATISTIC_CLS(d1), self.STATISTIC_CLS(d2)
        merged_algebraic_m = m1 | m2 - m2
        self._assert_equal(m1, merged_algebraic_m)

class MeanTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = Mean
    def _generate_data_set(self, size):
        return [random.randint(-1000, 1000) for i in range(size)]

    def _assert_equal(self, s1, s2):
        self.assertAlmostEqual(s1.get_mean(), s2.get_mean(), places=6)
        self.assertEqual(s1.get_n(), s2.get_n())

    def test_against_gold(self):
        dataset = self._generate_data_set(7)
        m = self.STATISTIC_CLS(dataset)
        true_mean = 1.0 * sum(dataset) / len(dataset)
        self.assertEqual(true_mean, m.get_mean())
        self.assertEqual(len(dataset), m.get_n())

class VectorMeanTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = Mean
    def _generate_data_set(self, size):
        return [np.random.uniform(-1000, 1000, size=2) for i in range(size)]

    def _assert_equal(self, s1, s2):
        self.assertTrue(self._almost_equal(s1.get_mean(), s2.get_mean()))
        self.assertEqual(s1.get_n(), s2.get_n())

    def _almost_equal(self, v1, v2):
        return np.all(np.isclose(v1, v2))

    def test_against_gold(self):
        dataset = self._generate_data_set(7)
        m = self.STATISTIC_CLS(dataset)
        true_mean = 1.0 * sum(dataset) / len(dataset)
        self.assertTrue(self._almost_equal(true_mean, m.get_mean()))
        self.assertEqual(len(dataset), m.get_n())

class VarianceTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = Variance
    def _generate_data_set(self, size):
        return [random.randint(-1000, 1000) for i in range(size)]

    def _assert_equal(self, s1, s2):
        self.assertAlmostEqual(s1.get_variance(), s2.get_variance(), places=6)

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

if __name__ == '__main__':
    unittest.main()