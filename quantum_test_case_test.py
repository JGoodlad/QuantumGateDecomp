import unittest

import numpy as np
import numpy.testing

from quantum_test_case import QuantumTestCase

class QuantumTestCaseTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.test_case = QuantumTestCase()

    def test_to_state(self):

        bit_string = '1010'

        state = self.test_case.to_state(bit_string)

        expected_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.complex)

        np.testing.assert_array_equal(state, expected_state)

if __name__ == '__main__': 
    unittest.main() 