import unittest
from tests.algebraic_statistic_tests import AbstractGroupStatisticTest
from conjugate_density import Bernoulli
from random import choice

class BernoulliTest(AbstractGroupStatisticTest):
    STATISTIC_CLS = Bernoulli
    def _generate_data_set(self, size):
        return [choice([0,1]) for i in range(size)]

    def _assert_equal(self, s1, s2):
        self.assertEqual(s1['Frequency'].get_counter(), s2['Frequency'].get_counter())

    def test_uninformative_prior(self):
        b = Bernoulli()
        self.assertAlmostEqual(b.get_posterior_pdf(0.0), b.get_posterior_pdf(1.0))
        self.assertAlmostEqual(b.get_posterior_pdf(0.0), b.get_posterior_pdf(0.5))
        self.assertAlmostEqual(b.get_posterior_pdf(0.4), b.get_posterior_pdf(0.5))
        self.assertAlmostEqual(b.get_posterior_pdf(0.3), b.get_posterior_pdf(0.7))

    def test_fair_count(self):
        data = [0]*100 + [1]*100
        b = Bernoulli(data)
        self.assertAlmostEqual(b.get_posterior_pdf(0.0), b.get_posterior_pdf(1.0))
        self.assertLess(b.get_posterior_pdf(0.0), b.get_posterior_pdf(0.5))
        self.assertAlmostEqual(b.get_posterior_pdf(0.3), b.get_posterior_pdf(0.7))

if __name__ == '__main__':
    unittest.main()