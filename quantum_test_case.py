import unittest 

from typing import Any, List

import cirq
import numpy as np

from abstract_qubit import CirqQubit, Qubit


  
class QuantumTestCase(unittest.TestCase):
    """Provides methods for convenient QC unit testing."""
    
    def setUp(self):
        pass

    def get_qubits(self, n:int) -> List[Qubit]:
        """Gets the specified number of qubits."""
        return [CirqQubit(i) for i in range(n)]

    def simulate(self, qubits: List[Qubit], gates: List[Any], initial_state, print_circuit=False):
        native_qubits = [q.native_qubit for q in qubits]

        circuit = cirq.Circuit()

        circuit.append(gates)

        if print_circuit:
            print(circuit)

        simulator = cirq.Simulator()

        simulator_result = simulator.simulate(circuit, qubit_order=native_qubits, initial_state=initial_state)

        final_state = simulator_result.final_state

        return final_state

    def to_state(self, bit_string: str) -> np.array:
        if not bit_string:
            raise ValueError('bit_string is unexpectedly falsy')

        char = bit_string[0]
        if char == '0':
            state = self.basis_0
        elif char == '1':
            state = self.basis_1
        else:
            raise ValueError('to_state expected only 1s and 0s')
        
        for char in bit_string[1:]:
            if char == '0':
                qubit = self.basis_0
            elif char == '1':
                qubit = self.basis_1
            else:
                raise ValueError('to_state expected only 1s and 0s')
            state = np.kron(state, qubit)
        
        return state
    
    def to_bit_string(self, n, num_qubits):
        binary_num = bin(n)[2:]
        padded = binary_num.zfill(num_qubits)
        return padded

    @property
    def basis_0(self):
        return np.array([1, 0], dtype=np.complex64)

    @property
    def basis_1(self):
        return np.array([0, 1], dtype=np.complex64)

