import cirq
import cirq.ops

import abstract_qubit
import primitives

class CirqGatePrimites(primitives.GatePrimitives):

    def H(self, qubit):
        return cirq.H(qubit.native_qubit)
    
    def X(self, qubit):
        return cirq.X(qubit.native_qubit)

    def CNOT(self, action_qubit, control_qubit):
        return cirq.CNOT(control_qubit.native_qubit, action_qubit.native_qubit)
    
    def CnNOT(self, action_qubit, *control_qubits):
        return self.CnU(cirq.ops.X._unitary_(), action_qubit, *control_qubits)
    
    def Rx(self, rads, qubit):
        return cirq.Rx(rads)(qubit.native_qubit)

    def Ry(self, rads, qubit):
        return cirq.Ry(rads)(qubit.native_qubit)
        
    def Rz(self, rads, qubit):
        return cirq.Rz(rads)(qubit.native_qubit)

    def U(self, unitary_matrix, qubit):
        return cirq.SingleQubitMatrixGate(unitary_matrix)(qubit.native_qubit)

    def CU(self, unitary_matrix, action_qubit, control_qubit):
        return cirq.ControlledGate(cirq.SingleQubitMatrixGate(unitary_matrix), [control_qubit.native_qubit])(action_qubit.native_qubit)

    def CnU(self, unitary_matrix, action_qubit, *control_qubits):
        native_qubits = [q.native_qubit for q in control_qubits]
        return cirq.ControlledGate(cirq.SingleQubitMatrixGate(unitary_matrix), native_qubits)(action_qubit.native_qubit)