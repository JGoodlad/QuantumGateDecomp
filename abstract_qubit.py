import abc

from typing import Any

import cirq


class Qubit(abc.ABC):

    @property
    @abc.abstractmethod
    def native_qubit(self) -> Any:
        """Gets the underlying qubit."""


class CirqQubit(Qubit):
    """A Cirq qubit."""

    def __init__(self, x_position: int):
        self.q = cirq.LineQubit(x_position)

    @property
    def native_qubit(self):
        return self.q