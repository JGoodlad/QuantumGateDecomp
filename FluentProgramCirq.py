import sys, os, time, math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cirq
from cirq.devices import GridQubit
from cirq.ops import CNOT, H, S, X, Z

import numpy as np

import utilCirq as util


class FluentProgramCirq(object):
    def __init__(self, n):
        self.definedGates = dict()
        self.program = cirq.Circuit()
        self.n = n
        self.qubits = [cirq.GridQubit(0, i) for i in range(n)]
    
    def get_program(self):
        return self.program
    
    def hadamard(self, *args):
        adds = []
        for arg in args:
            adds.append(cirq.H(self.qubits[arg]))
        self.program.append(adds, strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
        return self

    def s(self, *args):
        for arg in args:
            self.program.append(cirq.S(self.qubits[arg]))
        return self
    
    def cnot(self, a, b):
        self.program.append(cirq.CNOT(self.qubits[a], self.qubits[b]))
    
    def ccnot(self, a, b, c):
        self.program.append(cirq.TOFFOLI(self.qubits[a], self.qubits[b], self.qubits[c]))

    def x(self, *args):
        adds = []
        for arg in args:
            adds.append(cirq.X(self.qubits[arg]))
        self.program.append(adds, strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
        return self

    def z(self, *args):
        adds = []
        for arg in args:
            adds.append(cirq.Z(self.qubits[arg]))
        self.program.append(adds, strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
        return self
    
    def cnnot(self, *args):
        n = len(args)

        if n <= 1:
             raise Exception("ToffoliN requires at least 1 control bit and 1 output bit (i.e. n >= 2)")
        
        controlBits = list([self.qubits[i] for i in args[:-1]])
        tofN = cirq.ControlledGate(cirq.X, controlBits)

        self.program.append(tofN(self.qubits[args[-1]]))

        return self

    def measure_all(self):
        adds = []
        for i, v in enumerate(self.qubits):
            adds.append(cirq.measure(v, key=i))
        self.program.append(adds, strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
    
    def run_and_report(self, debug:bool = False):
        self.measure_all()
        simulator = cirq.Simulator()
        if debug:
            print(self.program)
        result = simulator.simulate(self.program, qubit_order=self.qubits)
        res = result.final_state
        print(res)
        return res