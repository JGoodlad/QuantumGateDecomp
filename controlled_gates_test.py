import unittest

import numpy as np
import numpy.testing

import controlled_gates
import gate_literals
import primitives_cirq
import quantum_test



class RecuriveControlledGateTest(quantum_test.QuantumTestCase):
    _MAX_QUBITS = 6

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
                        actual_final_state = self.simulate(qubits, actual_gates, initial_state)

                        expected_gates = [self.primitves.CnU(gate, qubits[-1], *qubits[:-1])]
                        expected_final_state = self.simulate(qubits, expected_gates, initial_state)

                        np.testing.assert_array_almost_equal_nulp(actual_final_state, expected_final_state)


if __name__ == '__main__':
    unittest.main()