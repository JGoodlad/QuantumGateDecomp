import os, sys, time

from typing import List

import numpy as np
import numpy.linalg

import scipy
import scipy.linalg

import sympy.combinatorics.graycode

from cirq import ControlledGate
from cirq.ops import CNOT, H, S, X, Z
from cirq.ops.matrix_gates import SingleQubitMatrixGate

import primitives

class RecuriveControlledGate(object):

    def __init__(self, gate_primitves: primitives.GatePrimitives):
        self.primitives: primitives.GatePrimitives = gate_primitves

    def _raise_expected_control_qubit_error(self, name: str):
        raise ValueError(f'{name} expected at least 1 control qubit')

    def _controlled_single_qubit_matrix_gate(self, matrix, action_qubit, control_qubit):
        return self.primitives.CU(matrix, action_qubit, control_qubit)

    def _controlled_n_not_gate(self, action_qubit, *control_qubits):
        '''Convenience function for Controlled-N NOT gate.'''
        return self._controlled_n_unitary_gate_recursive(X._unitary_(), action_qubit, *control_qubits)

    def _controlled_n_unitary_gate_recursive(self, matrix, action_bit, *control_bits):
        if len(control_bits) == 0:
            self._raise_expected_control_qubit_error('Controlled-N of U')

        if len(control_bits) == 1:
            return [self._controlled_single_qubit_matrix_gate(matrix, action_bit, control_bits[0])]

        # Aliases to make function calls more clear
        c_n_u = self._controlled_n_unitary_gate_recursive
        c_n_not = self._controlled_n_not_gate
        
        u = matrix
        v = scipy.linalg.sqrtm(u)
        v_h = v.conjugate().transpose()

        gates = []
        gates += c_n_u(v, action_bit, control_bits[-1])
        gates += c_n_not(control_bits[-1], *control_bits[:-1])
        gates += c_n_u(v_h, action_bit, control_bits[-1])
        gates += c_n_not(control_bits[-1], *control_bits[:-1])
        gates += c_n_u(v, action_bit, *control_bits[:-1])

        return gates

    # TODO: Enforce at least one control bit via action_qubit, control_1, *control_rest
    def CnU(self, unitary, action_qubit, *control_qubits):
        gates = self._controlled_n_unitary_gate_recursive(unitary, action_qubit, *control_qubits)
        # print(f'Gates used in construction: {len(gates)}')
        return gates

class IterativeControlledGate(object):
    def __init__(self, gate_primitves: primitives.GatePrimitives):
        self.primitives = gate_primitves

    def _get_operation_codes(self, n: int) -> List[str]:
        gray_code_generator = sympy.combinatorics.graycode.GrayCode(n).generate_gray()
        gray_codes = list(gray_code_generator)
        operation_codes = [c[::-1] for c in gray_codes]

        return operation_codes

    def _get_involved_qubits(self, operation_code, control_qubits):
        qubits = []
        for code_bit, qubit in zip(operation_code, control_qubits):
            if code_bit == '1':
                qubits += [qubit]
        return qubits
    
    def _get_last_involved_qubit(self, operation_code, control_qubits):
        for code_bit, qubit in zip(operation_code[::-1], control_qubits[::-1]):
            if code_bit == '1':
                return qubit
        pass

    def _get_dif_xored_qubits(self, current_opetation_code, previous_operation_code, control_qubits):
        def qubit_tuples(operation_code):
            qubits = self._get_involved_qubits(operation_code, control_qubits)
            qubit_tuples = []
            for q in qubits[:-1]:
                qubit_tuples += [(q, qubits[-1])]
            return qubit_tuples
        
        qubits = []
        tuples1, tuples2 = qubit_tuples(current_opetation_code), qubit_tuples(previous_operation_code)

        tl, ts = sorted([tuples1, tuples2], reverse=True, key=lambda x: len(x))

        change = list(set(tl) - set(ts))
        if not change:
            return []
        return list(*change) 


    def _controlled_n_unitary_gate_iterative(self, unitary, action_qubit, *control_bits):

        gates = []

        def xor(*qubits):
            nonlocal gates
            for q in qubits[:-1]:
                gates += [self.primitives.CNOT(qubits[-1], q)]

        def parity(operation_code: str):
            return operation_code.count('1') % 2 == 0

        def root(matrix, nth_power_of_two: int):
            for _ in range(nth_power_of_two):
                matrix = scipy.linalg.sqrtm(matrix)
            return matrix

        v = root(unitary, len(control_bits) - 1)
        v_h = v.conjugate().transpose()
        operations = self._get_operation_codes(len(control_bits))

        prev_op = operations[0]
        for curr_op in operations[1:]:
            # Get the different xored qubits
            xored_qubits = self._get_dif_xored_qubits(curr_op, prev_op, control_bits)

            # Apply xor
            xor(*xored_qubits)

            last_involved_qubit = self._get_last_involved_qubit(curr_op, control_bits)

            if not parity(curr_op):
                # Odd; apply normal
                gates += [self.primitives.CU(v, action_qubit, last_involved_qubit)]
            else:
                # Even; apply inverse
                gates += [self.primitives.CU(v_h, action_qubit, last_involved_qubit)]
            prev_op = curr_op
        
        program_ordered_gates = gates[::-1]
        return program_ordered_gates


    def CnU(self, unitary, action_qubit, *control_qubits):
        gates = self._controlled_n_unitary_gate_iterative(unitary, action_qubit, *control_qubits)
        return gates

    


class ElementaryComposedGate(object):

    def __init__(self, gate_primitves: primitives.GatePrimitives):
        self.primitives = gate_primitves

    def _assert_det_unitary(self, unitary_matrix):
        det = np.linalg.det(unitary_matrix)
        if not np.isclose(det, 1):
            raise AssertionError('Expected unitary determinant.')

    def _assert_close(self, actual, expected):
        if not np.allclose(actual, expected):
            raise AssertionError('Actual value not close to expected')

    def U(self, unitary_matrix, qubit):
        g, a, b, t = self._gamma_alpha_beta_theta_factorization(unitary_matrix)

        gates = []
        
        gates += [self.primitives.Ph(g, qubit)]
        gates += [self.primitives.Rz(-a, qubit)]
        gates += [self.primitives.Ry(-t, qubit)]
        gates += [self.primitives.Rz(-b, qubit)]

        program_ordered_gates = gates[::-1]
        return program_ordered_gates

    def CU(self, unitary_matrix, action_qubit, control_qubit):
        g, a, b, t = self._gamma_alpha_beta_theta_factorization(unitary_matrix)
        
        gates = []
        
        def A():
            nonlocal gates
            gates += [self.primitives.Rz(-a, action_qubit)]
            gates += [self.primitives.Ry(-t/2, action_qubit)]

        def B():
            nonlocal gates
            gates += [self.primitives.Ry(t/2, action_qubit)]
            gates += [self.primitives.Rz((a + b)/ 2, action_qubit)]

        def C():
            nonlocal gates
            gates += [self.primitives.Rz(-1 * (b - a)/2, action_qubit)]
        
        def CPh():
            nonlocal gates
            gates += [self.primitives.Rz(g, control_qubit)]
            gates += [self.primitives.Ph(g/2, control_qubit)]

        CPh()
        A()
        gates += [self.primitives.CNOT(action_qubit, control_qubit)]
        B()
        gates += [self.primitives.CNOT(action_qubit, control_qubit)]
        C()

        program_ordered_gates = gates[::-1]
        return program_ordered_gates

    
    def _gamma_alpha_beta_theta_factorization(self, unitary_matrix):
        # copy matrix to avoid changing it
        unitary_matrix = np.copy(unitary_matrix)

        det = np.linalg.det(unitary_matrix)
        gamma = np.angle(np.sqrt(det))
        special_unitary_matrix = 1 / np.sqrt(det) * unitary_matrix
        self._assert_det_unitary(special_unitary_matrix)

        u00, u01  = special_unitary_matrix[0][[0, 1]]

        a, b = np.angle(u00), np.angle(u01)
        # a = alpha + beta and b = alpha -  beta
        alpha = (a + b) / 2
        beta = (a - b) / 2
        theta = np.arccos(np.abs(u00))

        return gamma, alpha * 2, beta * 2, theta * 2
