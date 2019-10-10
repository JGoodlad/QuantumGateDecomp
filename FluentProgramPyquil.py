import sys, os, time, math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


import pyquil
import pyquil.gates
import pyquil.api

import Pyquil.utilPyquil as util


class FluentProgramPyquil(object):
    def __init__(self, n, program_out_callback=None):
        self.program_out_callback = program_out_callback
        self.definedGates = dict()
        self.program = pyquil.Program()
        self.n = n
        self.qubits = [0, 1, 2, 6, 7, 10, 11, 13, 14, 15, 16, 17]
    
    def GetProgram(self):
        return self.program
    
    def HadamardN(self, *args):
        for arg in args:
            self.program += pyquil.gates.H(self.qubits[arg])
        return self

    def SN(self, *args):
        for arg in args:
            self.program += pyquil.gatest.S(self.qubits[arg])
        return self
    
    def Cnot(self, a, b):
        self.program += pyquil.gates.CNOT(self.qubits[a], self.qubits[b])
    
    def Ccnot(self, a, b, c):
        self.program += pyquil.gates.TOFFOLI(self.qubits[a], self.qubits[b], self.qubits[c])

    def XN(self, *args):
        for arg in args:
            self.program += pyquil.gates.X(self.qubits[arg])
        return self
    
    def ZN(self, *args):
        for arg in args:
            self.program += pyquil.gates.Z(self.qubits[arg])
        return self
    
    def ToffoliN(self, *args):
        n = len(args)

        if n <= 1:
             raise Exception("ToffoliN requires at least 1 control bit and 1 output bit (i.e. n >= 2)")
        
        controlBits = list([self.qubits[i] for i in args[:-1]])
        t = pyquil.gates.X(self.qubits[args[-1]])
        for i in controlBits:
            t = t.controlled(i)

        self.program += t

        return self

    def MeasureAll(self):
        pass
    
    def RunAndReport(self, trials:int = 1, debug:bool = False):
        self.MeasureAll()
        qvm = pyquil.api.get_qc(util.get_qc_name(), as_qvm=True)
        qvm.compiler.client.timeout = 600
        with pyquil.api.local_qvm():
            if self.program_out_callback:
                code = self.program 
                #code = qvm.compiler.quil_to_native_quil(self.program)
                self.program_out_callback(code)
                util.throw_if_set()
            if debug:
                print(self.program)
            results = qvm.run_and_measure(self.program, trials=trials)
            return util.extractResultsForQubits(results, *list([self.qubits[i] for i in range(0, self.n)]))

        # if debug:
        #     print(self.program)
        # result = simulator.run(self.program, repetitions=t)
        # res = util.extractResultsForQubits(result.measurements, *list(range(0, self.n)))
        # return res

    def RunAndReport2(self, trials = 1, mes = None, debug:bool = False):
        self.MeasureAll()
        qvm = pyquil.api.get_qc(util.get_qc_name(), as_qvm=True)
        qvm.compiler.client.timeout = 600
        with pyquil.api.local_qvm():
            if self.program_out_callback:
                #code = self.program
                code = qvm.compiler.quil_to_native_quil(self.program)
                self.program_out_callback(code)
                util.throw_if_set()
            if debug:
                print(self.program)
            results = qvm.run_and_measure(self.program, trials=trials)
            if not mes:
                mes = list([self.qubits[i] for i in range(0, self.n)])
            return util.extractResultsPerRunForQubits(results, *mes)
