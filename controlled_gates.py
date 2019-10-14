import os, sys, time

import numpy as np
import numpy.linalg

from scipy import linalg

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
        v = linalg.sqrtm(u)
        v_h = v.conjugate().transpose()

        gates = []
        gates += c_n_u(v, action_bit, control_bits[-1])
        gates += c_n_not(control_bits[-1], *control_bits[:-1])
        gates += c_n_u(v_h, action_bit, control_bits[-1])
        gates += c_n_not(control_bits[-1], *control_bits[:-1])
        gates += c_n_u(v, action_bit, *control_bits[:-1])

        return gates

    # TODO: Enforce at least one control bit via action_qubit, control_1, *control_rest
    def controlled_n_unitary_gate(self, unitary, action_qubit, *control_qubits):
        gates = self._controlled_n_unitary_gate_recursive(unitary, action_qubit, *control_qubits)
        # print(f'Gates used in construction: {len(gates)}')
        return gates


class ElementaryUControlledGate(object):

    def __init__(self, gate_primitves: primitives.GatePrimitives):
        self.primitives = gate_primitves

    def _assert_det_unitary(self, unitary_matrix):
        det = np.linalg.det(unitary_matrix)
        if not np.isclose(det, 1):
            raise AssertionError('Expected unitary determinant.')

    def U(self, unitary_matrix, qubit):
        gates = []

        det = np.linalg.det(unitary_matrix)

        # Correction to make det = 1
        # TODO: Understand why preappend the conj of phase, not phase
        if not np.isclose(det, 1):
            phase = np.sqrt(1/det) * np.identity(2)

            unitary_matrix = phase @ unitary_matrix
            gates += [self.primitives.U(phase.conj(), qubit)]

            self._assert_det_unitary(unitary_matrix)

        q11 = unitary_matrix[0][0]
        q21 = unitary_matrix[0][1]

        _theta = np.arccos(np.abs(q11))
        _alpha = np.angle(q11)
        _beta = np.angle(q21)

        
        m1, m2, m3 = self.U_as_phase_rotation_phase(_alpha, _beta, _theta)

        # TODO: Understand why m3 is the first, it seems it should be m1
        gates += [self.primitives.U(m1, qubit)]
        gates += [self.primitives.U(m2, qubit)]
        gates += [self.primitives.U(m3, qubit)]

        return gates
    
    def U_as_rotation(self, a, b, t):
        a1 = [np.exp(1j * a) * np.cos(t), np.exp(1j * b) * np.sin(t)]
        a2 = [-1 * np.exp(1j * -b) * np.sin(t), np.exp(1j * -a) * np.cos(t)]
        return np.array([a1, a2])

    def U_as_phase_rotation_phase(self, a, b, t):
        # φ1 = ψ + Δ and φ2 = ψ − Δ
        # p1 - p2 = 2d
        psi = (a + b) / 2
        delta = (a - b) / 2
        m1 = self.phase(delta)
        m2 = self.rotation(t)
        m3 = self.phase(psi)

        return m1, m2, m3

    def rotation(self, x):
        row1 = [np.cos(x), np.sin(x)]
        row2 = [-np.sin(x), np.cos(x)]
        return np.array([row1, row2])

    def phase(self, x):
        row1 = [np.exp(+1j * x), 0]
        row2 = [0, np.exp(-1j * x)]
        return np.array([row1, row2])

    # def C1U(self, unitary_matrix, action_qubit, control_qubit):
    #     pass

    def U_1_gate(self, unitary_matrix, qubit):
        gates = []

        det = np.linalg.det(unitary_matrix)

        # Correction to make det = 1
        # TODO: Understand why preappend the conj of phase, not phase
        if not np.isclose(det, 1):
            phase = np.sqrt(1/det) * np.identity(2)

            unitary_matrix = phase @ unitary_matrix
            gates += [self.primitives.U(phase.conj(), qubit)]

            self._assert_det_unitary(unitary_matrix)

        q11 = unitary_matrix[0][0]
        q21 = unitary_matrix[0][1]

        _theta = np.arccos(np.abs(q11))
        _alpha = np.angle(q11)
        _beta = np.angle(q21)

        
        matrix = self.U_as_rotation(_alpha, _beta, _theta)

        gates += [self.primitives.U(matrix, qubit)]


        # alpha = np.pi / 2
        # beta = np.pi * 7 / 2
        # theta = 2 * np.arccos(abs(q11))

        # gates = []
        # gates += [self.primitives.Rz(alpha, qubit)]
        # gates += [self.primitives.Ry(theta, qubit)]
        # gates += [self.primitives.Rz(beta, qubit)]

        return gates