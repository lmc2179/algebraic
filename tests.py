import unittest
from algebraic_statistic import Mean, Variance, NormalEstimator
import random

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

    def test_equality(self):
        d1, d2 = self._generate_data_sets([3, 4])
        self.assertTrue(self.STATISTIC_CLS(d1) == self.STATISTIC_CLS(d1))
        self.assertFalse(self.STATISTIC_CLS(d1) == self.STATISTIC_CLS(d2))

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

class VarianceTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = Variance
    def _generate_data_set(self, size):
        return [random.randint(-1000, 1000) for i in range(size)]

    def _assert_equal(self, s1, s2):
        self.assertAlmostEqual(s1.get_variance(), s2.get_variance(), places=6)

class NormalEstimatorTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = NormalEstimator
    def _generate_data_set(self, size):
        return [random.randint(-1000, 1000) for i in range(size)]

    def _assert_equal(self, s1, s2):
        self.assertAlmostEqual(s1.statistic_values['variance'].get_variance(),
                               s2.statistic_values['variance'].get_variance(), places=6)
        self.assertAlmostEqual(s1.statistic_values['mean'].get_mean(),
                               s2.statistic_values['mean'].get_mean(), places=6)

if __name__ == '__main__':
    unittest.main()