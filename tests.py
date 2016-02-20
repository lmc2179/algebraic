import unittest
from algebraic_statistic import Mean

class MeanTest(unittest.TestCase):
    def test_mean_calculation_correctness(self):
        dataset = [1, 3, 4, -10, 2334, 3, 100]
        m = Mean(dataset)
        true_mean = 1.0 * sum(dataset) / len(dataset)
        self.assertEqual(true_mean, m.get_mean())
        self.assertEqual(len(dataset), m.get_n())

    def test_mean_merge_correctness(self):
        d1, d2 = [1, 3, 4] ,[-10, 2334, 3, 100]
        merged_dataset = d1 + d2
        merged_dataset_m = Mean(merged_dataset)
        m1, m2 = Mean(d1), Mean(d2)
        merged_algebraic_m = m1 | m2
        self.assertEqual(merged_dataset_m.get_mean(), merged_algebraic_m.get_mean())
        self.assertEqual(merged_dataset_m.get_n(), merged_algebraic_m.get_n())

    def test_mean_eq(self):
        d1, d2 = [1, 3, 4] ,[-10, 2334, 3, 100]
        self.assertTrue(Mean(d1) == Mean(d1))
        self.assertFalse(Mean(d1) == Mean(d2))

    def test_inverse(self):
        d1, d2 = [1, 3, 4] ,[-10, 2334, 3, 100]
        m1, m2 = Mean(d1), Mean(d2)
        merged_algebraic_m = m1 | m2 | -m2
        self.assertEqual(m1.get_mean(), merged_algebraic_m.get_mean())
        self.assertEqual(m1.get_n(), merged_algebraic_m.get_n())

    def test_inverse_to_identity(self):
        dataset = [1, 3, 4, -10, 2334, 3, 100]
        m = Mean(dataset) - Mean(dataset)
        m_ident = Mean.get_identity()
        self.assertEqual(m.get_mean(), m_ident.get_mean())
        self.assertEqual(m.get_n(), m_ident.get_n())

    def test_sub(self):
        d1, d2 = [1, 3, 4] ,[-10, 2334, 3, 100]
        m1, m2 = Mean(d1), Mean(d2)
        merged_algebraic_m = m1 | m2 - m2
        self.assertEqual(m1.get_mean(), merged_algebraic_m.get_mean())
        self.assertEqual(m1.get_n(), merged_algebraic_m.get_n())

    def test_identity(self):
        dataset = [1, 3, 4, -10, 2334, 3, 100]
        m = Mean(dataset) | Mean.get_identity()
        true_mean = 1.0 * sum(dataset) / len(dataset)
        self.assertEqual(true_mean, m.get_mean())
        self.assertEqual(len(dataset), m.get_n())

    def test_associativity(self):
        d1, d2, d3 = [1, 3, 4] ,[-10, 2334, 3, 100], [-10, 20, 2200,3,3,4]
        m1, m2, m3 = Mean(d1), Mean(d2), Mean(d3)
        merged_algebraic_m_left = (m1 | m2) | m3
        merged_algebraic_m_right = m1 | (m2 | m3)
        self.assertEqual(merged_algebraic_m_left.get_mean(), merged_algebraic_m_right.get_mean())
        self.assertEqual(merged_algebraic_m_left.get_n(), merged_algebraic_m_right.get_n())

if __name__ == '__main__':
    unittest.main()