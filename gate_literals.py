'''
This module provides primitive gates as their unitary matricies.

NOTE: This module *should not* be directly imported
'''

import cirq
import cirq.ops



X = cirq.ops.X._unitary_()
Y = cirq.ops.Y._unitary_()
Z = cirq.ops.Z._unitary_()
Rx = lambda theta: cirq.ops.Rx(theta)._unitary_()
Ry = lambda theta: cirq.ops.Ry(theta)._unitary_()
Rz = lambda theta: cirq.ops.Rz(theta)._unitary_()