import abc

class GatePrimitives(abc.ABC):

    @abc.abstractmethod
    def H(self, qubit):
        pass
    
    @abc.abstractmethod
    def X(self, qubit):
        pass

    @abc.abstractmethod
    def CNOT(self, action_qubit, control_qubit):
        pass
    
    @abc.abstractmethod
    def CnNOT(self, action_qubit, *control_qubits):
        pass
    
    @abc.abstractmethod
    def Rx(self, rads, qubit):
        pass

    @abc.abstractmethod
    def Ry(self, rads, qubit):
        pass
    
    @abc.abstractmethod
    def Rz(self, rads, qubit):
        pass

    @abc.abstractmethod
    def U(self, unitary_matrix, qubit):
        pass

    @abc.abstractmethod
    def CU(self, unitary_matrix, action_qubit, control_qubit):
        pass

    @abc.abstractmethod
    def CnU(self, unitary_matrix, action_qubit, *control_qubits):
        pass


