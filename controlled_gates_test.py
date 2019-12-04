import time

import unittest

import numpy as np
import numpy.testing

import controlled_gates
import gate_literals
import primitives_cirq
import quantum_test


class ControlledGateTest(quantum_test.QuantumTestCase):
    _MAX_QUBITS = 5

    def setUp(self):
        super().setUp()

        self.primitves = primitives_cirq.CirqGatePrimites()

        theta = np.pi / 4
        self.gates_to_test = [
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

        self.named_gates_to_test = [
            ('X^(1/2)',  gate_literals.SQRT_X),
            ('X^(1/4)', gate_literals.SQRT_(gate_literals.SQRT_X)),
            ('X^(1/8)', gate_literals.SQRT_(gate_literals.SQRT_(gate_literals.SQRT_X))),
            ('H', gate_literals.H),
            ('X', gate_literals.X),
            ('Y', gate_literals.Y), 
            ('Z', gate_literals.Z), 
            ('Rx', gate_literals.Rx(theta)), 
            ('Ry', gate_literals.Ry(theta)),
            ('Rz', gate_literals.Rz(theta))
        ]

    def test_IterativeControlledGate_CnU(self):
        gate_builder = controlled_gates.IterativeControlledGate(self.primitves)

        for n in range(2, self._MAX_QUBITS + 1):
            qubits = self.get_qubits(n)
            for initial_state_number in range(0, 2**n):
                for gate in self.gates_to_test:
                    with self.subTest(num_qubits=n, initial_state_number=initial_state_number, gate=gate):

                        initial_bit_string = self.to_bit_string(initial_state_number, n)
                        initial_state = self.to_state(initial_bit_string)

                        actual_gates = gate_builder.CnU(gate, qubits[-1], *qubits[:-1])
                        actual_final_state = self.simulate(qubits, actual_gates, np.copy(initial_state))

                        expected_gates = [self.primitves.CnU(gate, qubits[-1], *qubits[:-1])]
                        expected_final_state = self.simulate(qubits, expected_gates, np.copy(initial_state))

                        np.testing.assert_almost_equal(actual_final_state, expected_final_state, decimal=5)


    def test_RecursiveControlledGate_CnU(self):

        gate_builder = controlled_gates.RecuriveControlledGate(self.primitves)

        for n in range(2, self._MAX_QUBITS + 1):
            qubits = self.get_qubits(n)
            for initial_state_number in range(0, 2**n):
                for gate in self.gates_to_test:
                    with self.subTest(num_qubits=n, initial_state_number=initial_state_number, gate=gate):

                        initial_bit_string = self.to_bit_string(initial_state_number, n)
                        initial_state = self.to_state(initial_bit_string)

                        actual_gates = gate_builder.CnU(gate, qubits[-1], *qubits[:-1])
                        actual_final_state = self.simulate(qubits, actual_gates, np.copy(initial_state))

                        expected_gates = [self.primitves.CnU(gate, qubits[-1], *qubits[:-1])]
                        expected_final_state = self.simulate(qubits, expected_gates, np.copy(initial_state))

                        np.testing.assert_almost_equal(actual_final_state, expected_final_state, decimal=5)

    def test_ElementarilyComposedGates_U(self):
        gate_builder = controlled_gates.ElementaryComposedGate(self.primitves)

        [qubit] = self.get_qubits(1)
        for gate in self.gates_to_test:
            for initial_state_number in range(0, 2):
                with self.subTest(initial_state_number=initial_state_number, gate=gate):
                    initial_bit_string = self.to_bit_string(initial_state_number, 1)
                    initial_state = self.to_state(initial_bit_string)

                    actual_gates = gate_builder.U(gate, qubit)
                    actual_final_state = self.simulate([qubit], actual_gates, np.copy(initial_state))

                    expected_gates = [self.primitves.U(gate, qubit)]
                    expected_final_state = self.simulate([qubit], expected_gates, np.copy(initial_state))

                    np.testing.assert_almost_equal(actual_final_state, expected_final_state, decimal=5)

    def test_ElementarilyComposedGates_CU(self):
        gate_builder = controlled_gates.ElementaryComposedGate(self.primitves)

        qubits = self.get_qubits(2)
        for gate in self.gates_to_test:
            for initial_state_number in range(0, 4):
                with self.subTest(initial_state_number=initial_state_number, gate=gate):
                    initial_bit_string = self.to_bit_string(initial_state_number, 2)
                    initial_state = self.to_state(initial_bit_string)

                    actual_gates = gate_builder.CU(gate, *qubits[::-1])
                    actual_final_state = self.simulate(qubits, actual_gates, np.copy(initial_state))

                    expected_gates = [self.primitves.CU(gate, *qubits[::-1])]
                    expected_final_state = self.simulate(qubits, expected_gates, np.copy(initial_state))

                    np.testing.assert_almost_equal(actual_final_state, expected_final_state, decimal=5)

    def test_GateTiming_Single(self):
        recursive_gate_builder = controlled_gates.RecuriveControlledGate(self.primitves)
        iterative_gate_builder = controlled_gates.IterativeControlledGate(self.primitves)

        recursive_CN = lambda *x: recursive_gate_builder.CnU(*x)
        iterative_CN = lambda *x: iterative_gate_builder.CnU(*x)

        out_file = open('out_file_single.csv', 'w')

        print('name,gate_name,n,initial_state_number,construction_time,simulation_time,precision,abs_error,euclid_error', file=out_file)
        for name, CN in [('rec', recursive_CN), ('itr', iterative_CN)]:
            for n in range(2, self._MAX_QUBITS + 1):
                qubits = self.get_qubits(n)
                for initial_state_number in range(0, 2**n):
                    for gate_name, gate in self.named_gates_to_test:
                        initial_bit_string = self.to_bit_string(initial_state_number, n)
                        initial_state = self.to_state(initial_bit_string)

                        initial_bit_string = self.to_bit_string(initial_state_number, n)
                        initial_state = self.to_state(initial_bit_string)

                        construction_time, actual_gates = self.elapsed_time(lambda: CN(gate, qubits[-1], *qubits[:-1]))
                        simulation_time, actual_final_state = self.elapsed_time(lambda: self.simulate(qubits, actual_gates, np.copy(initial_state)))

                        expected_gates = [self.primitves.CnU(gate, qubits[-1], *qubits[:-1])]
                        expected_final_state = self.simulate(qubits, expected_gates, np.copy(initial_state))

                        precision = self.get_precision(actual_final_state, expected_final_state)
                        abs_error = self.get_max_absolute_error(actual_final_state, expected_final_state)
                        euclid_error = self.get_euclidean_error(actual_final_state, expected_final_state)

                        print(name, gate_name, n, initial_state_number, construction_time, simulation_time, precision, abs_error, euclid_error,
                        sep=',', file=out_file, flush=True)
        out_file.close()
        

    def test_GateTiming_Aggrigated(self):
        recursive_gate_builder = controlled_gates.RecuriveControlledGate(self.primitves)
        iterative_gate_builder = controlled_gates.IterativeControlledGate(self.primitves)

        recursive_CN = lambda *x: recursive_gate_builder.CnU(*x)
        iterative_CN = lambda *x: iterative_gate_builder.CnU(*x)

        out_file = open('out_file_agg.csv', 'w')

        print('name,gate_name,n,np.average(construction_times),np.average(simulation_times),np.min(precisions),np.average(abs_errors),np.average(euclid_errors)', file=out_file)
        
        for n in range(2, self._MAX_QUBITS + 1):
            for name, CN in [('rec', recursive_CN), ('itr', iterative_CN)]:
                qubits = self.get_qubits(n)
                for gate_name, gate in self.named_gates_to_test:
                    construction_times = []
                    simulation_times = []
                    precisions = []
                    abs_errors = []
                    euclid_errors = []
                    for initial_state_number in range(0, 2**n):
                        initial_bit_string = self.to_bit_string(initial_state_number, n)
                        initial_state = self.to_state(initial_bit_string)

                        initial_bit_string = self.to_bit_string(initial_state_number, n)
                        initial_state = self.to_state(initial_bit_string)

                        construction_time, actual_gates = self.elapsed_time(lambda: CN(gate, qubits[-1], *qubits[:-1]))
                        construction_times += [construction_time]
                        simulation_time, actual_final_state = self.elapsed_time(lambda: self.simulate(qubits, actual_gates, np.copy(initial_state)))
                        simulation_times += [simulation_time]

                        expected_gates = [self.primitves.CnU(gate, qubits[-1], *qubits[:-1])]
                        expected_final_state = self.simulate(qubits, expected_gates, np.copy(initial_state))

                        precisions += [self.get_precision(actual_final_state, expected_final_state)]
                        abs_errors += [self.get_max_absolute_error(actual_final_state, expected_final_state)]
                        euclid_errors += [self.get_euclidean_error(actual_final_state, expected_final_state)]

                    print(name, gate_name, n, np.average(construction_times), np.average(simulation_times), np.amin(precisions), np.average(abs_errors), np.average(euclid_errors),
                    sep=',', file=out_file, flush=True)
        out_file.close()

    def elapsed_time(self, f):
        start = time.time()
        ret = f()
        end = time.time()
        elapsed = int(round((end - start) * 1000))
        return elapsed, ret


if __name__ == '__main__':
    unittest.main()