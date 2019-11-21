import unittest

import numpy as np
import numpy.testing

from quantum_test import QuantumTestCase

class QuantumTestCaseTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.test_case = QuantumTestCase()

    def test_to_state(self):

        bit_string = '1010'

        state = self.test_case.to_state(bit_string)

        expected_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.complex)

        np.testing.assert_array_equal(state, expected_state)

    def test_get_precision(self):
        pi = np.array([3.14159265])
        almost_pi = np.array([3.14160000])
        # NOTE: This test case has 4 digits that are the same, 
        # but 5 digits differ by only the ones place
        expected_precision = 5

        precision = self.test_case.get_precision(almost_pi, pi)

        self.assertEqual(precision, expected_precision)

    def test_get_max_aboslute_error(self):
        actaul, expected = np.array([0, 1]), np.array([1, 1])
        expected_max_absolute_error = 1

        max_absolute_error = self.test_case.get_max_absolute_error(actaul, expected)

        self.assertEqual(max_absolute_error, expected_max_absolute_error)
        
    def test_get_euclidean_error(self):
        actaul = np.array([1, 2])
        expected = np.array([4, 6])
        expected_euclidian_error = 5

        euclidian_error = self.test_case.get_euclidean_error(actaul, expected)

        self.assertEqual(euclidian_error, expected_euclidian_error)


if __name__ == '__main__': 
    unittest.main() 