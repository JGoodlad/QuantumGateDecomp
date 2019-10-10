import sys, os, time

import numpy as np

import quantum_test_case

from FluentProgramCirq import FluentProgramCirq
from utilCirq import getAll1Indexes, toBitString


from cirq import ControlledGate
from cirq.ops import CNOT, H, S, X, Z
from cirq.ops.matrix_gates import SingleQubitMatrixGate

from scipy import linalg

import gates_cirq


class RecuriveControlledGate(object):

    def __init__(self):
        self.primitives = gates_cirq.GatesCirq()
        pass

    def _raise_expected_control_qubit_error(self, name):
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



class TestDriver(object):
    def TestToffoliNative(self, n: int, inputString:str) -> bool:
        qt = quantum_test_case.QuantumTestCase()

        qubits = qt.get_qubits(n)

        initial_state = qt.to_state(inputString)

        gates = RecuriveControlledGate().controlled_n_unitary_gate(
            X._unitary_(), qubits[-1], *qubits[:-1])

        final_state = qt.simulate(qubits, gates, initial_state)

        expected_gates = [gates_cirq.GatesCirq().CnNOT(qubits[-1], *qubits[:-1])]
        expected_final_state = qt.simulate(qubits, expected_gates, initial_state)

        isPass = np.array_equal(final_state, expected_final_state)

        print(f'{"Passed:" if isPass else "FAILED:"} ToffoliNative Returned: {final_state}, Expected: {expected_final_state}, Input: {inputString}')
        return isPass


if __name__ == "__main__":
    for n in range(2, 4):
        t = []
        tests = 2**n
        for i in range(tests - 4, tests):
            s = ("{0:0"+ str(n) + "b}").format(i)
            if s[-1] != '0':
                continue
            start = time.time()
            res = TestDriver().TestToffoliNative(n, s)
            if not res:
                raise Exception("Not Correct.")

            end = time.time()
            curr = end-start
            t.append(curr)
        if t:
            print(sum(t)/len(t))