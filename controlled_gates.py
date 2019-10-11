import os, sys, time

import numpy as np

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
