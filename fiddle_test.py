import unittest

import numpy as np
import numpy.linalg

import scipy
import scipy.linalg

import gate_literals
import quantum_test


class FiddleTest(quantum_test.QuantumTestCase):
    n_max = 12

    def test_SigFigsNthRoot(self):
        def root(matrix, nth_power_of_two: int):
            for _ in range(nth_power_of_two):
                matrix = scipy.linalg.sqrtm(matrix)
            return matrix

        def noisey_mult(acc, matrix):
            return acc @ matrix
        
        base_matrix = gate_literals.X

        out_file = open('out_file_fiddle_2x2nthroot.csv', 'w')

        for n in range(self.n_max):
            nth_root_matrix = root(base_matrix, n)

            actual_matrix = np.identity(2)
            for i in range(2**n):
                actual_matrix = actual_matrix @ nth_root_matrix

            equal_sig_figs = self.get_precision(actual_matrix, base_matrix)
            max_abs_err = self.get_max_absolute_error(actual_matrix, base_matrix)

            print(equal_sig_figs)
            print(max_abs_err)
            print(f'{n},{equal_sig_figs},{max_abs_err}', file=out_file)

        out_file.close()

    def test_SigFigsNthRootControl(self):
        def root(matrix, nth_power_of_two: int):
            for _ in range(nth_power_of_two):
                matrix = scipy.linalg.sqrtm(matrix)
            return matrix

        def noisey_mult(acc, matrix):
            return acc @ matrix
        
        def make_controlled(unitary_matrix, num_control):
            base_matrix = np.identity(2**num_control, dtype=np.complex128)
            base_matrix[-1][-1] = unitary_matrix[-1][-1]
            base_matrix[-2][-2] = unitary_matrix[-2][-2]
            base_matrix[-1][-2] = unitary_matrix[-1][-2]
            base_matrix[-2][-1] = unitary_matrix[-2][-1]
            return base_matrix

        out_file = open('out_file_fiddle_nxnnthroot.csv', 'w')


        for n in range(1, self.n_max):
            base_matrix = make_controlled(gate_literals.X, n)
            nth_root_matrix = root(base_matrix, n)

            actual_matrix = np.identity(2**n)
            for i in range(2**n):
                actual_matrix = actual_matrix @ nth_root_matrix

            equal_sig_figs = self.get_precision(actual_matrix, base_matrix)
            max_abs_err = self.get_max_absolute_error(actual_matrix, base_matrix)

            print(equal_sig_figs)
            print(max_abs_err)
            print(f'{n},{equal_sig_figs},{max_abs_err}', file=out_file)

        out_file.close()


if __name__ == '__main__':
    unittest.main()