import unittest

import numpy as np
import numpy.testing

import controlled_gates
import gate_literals
import primitives_cirq
import quantum_test

import cirq
import cirq.ops


class RecuriveControlledGateTest(quantum_test.QuantumTestCase):
    _MAX_QUBITS = 3

    def setUp(self):
        super().setUp()

        self.primitves = primitives_cirq.CirqGatePrimites()

    def test_controlled_n_unitary_gate(self):

        gate_builder = controlled_gates.RecuriveControlledGate(self.primitves)
        theta = np.pi / 4
        gates_to_test = [
            gate_literals.X, 
            gate_literals.Y, 
            gate_literals.Z, 
            gate_literals.Rx(theta), 
            gate_literals.Ry(theta),
            gate_literals.Rz(theta)
        ]

        for n in range(2, self._MAX_QUBITS + 1):
            qubits = self.get_qubits(n)
            for initial_state_number in range(0, 2**n):
                for gate in gates_to_test:
                    with self.subTest(num_qubits=n, initial_state_number=initial_state_number, gate=gate):

                        initial_bit_string = self.to_bit_string(initial_state_number, n)
                        initial_state = self.to_state(initial_bit_string)

                        actual_gates = gate_builder.controlled_n_unitary_gate(gate, qubits[-1], *qubits[:-1])
                        actual_final_state = self.simulate(qubits, actual_gates, np.copy(initial_state))

                        expected_gates = [self.primitves.CnU(gate, qubits[-1], *qubits[:-1])]
                        expected_final_state = self.simulate(qubits, expected_gates, np.copy(initial_state))

                        np.testing.assert_almost_equal(actual_final_state, expected_final_state)

    def test_ElementarilyComposedGates_U(self):
        gate_builder = controlled_gates.ElementarilyComposedGates(self.primitves)
        theta = np.pi / 4
        gates_to_test = [
            gate_literals.SQRT_X,
            gate_literals.SQRT_(gate_literals.SQRT_X),
            gate_literals.SQRT_(gate_literals.SQRT_(gate_literals.SQRT_X)),
            gate_literals.H,
            gate_literals.X,
            gate_literals.Y, 
            gate_literals.Z, 
            gate_literals.Rx(theta), 
            gate_literals.Ry(theta),
            gate_literals.Rz(theta)
        ]

        [qubit] = self.get_qubits(1)
        for gate in gates_to_test:
            for initial_state_number in range(0, 2):
                with self.subTest(initial_state_number=initial_state_number, gate=gate):
                    initial_bit_string = self.to_bit_string(initial_state_number, 1)
                    initial_state = self.to_state(initial_bit_string)

                    actual_gates = gate_builder.U(gate, qubit)
                    actual_final_state = self.simulate([qubit], actual_gates, np.copy(initial_state))

                    expected_gates = [self.primitves.U(gate, qubit)]
                    expected_final_state = self.simulate([qubit], expected_gates, np.copy(initial_state))

                    np.testing.assert_almost_equal(actual_final_state, expected_final_state)


if __name__ == '__main__':
    unittest.main()