"The module for runtime exceptions."

import textwrap
from abc import ABCMeta
from dataclasses import dataclass
from typing import Generic, TypeVar

from qiskit.quantum_info import Statevector
from qiskit.result.result import Result

R = TypeVar("R")  # Type for Param
P = TypeVar("P")  # Type for Param


class DbCQuantumError(Exception):
    """
    The base class for exceptions raised by DbCQuantum.
    """

    pass


@dataclass
class ConditionErrorInfo(metaclass=ABCMeta):
    """
    The base class that holds information related to an assertion failure.
    """

    def __str__(self) -> str:
        ret: str = "[Info]"
        for instance_var_name in vars(self):
            ret += "\n"
            value_repr: str = repr(getattr(self, instance_var_name))
            if "\n" in value_repr:
                ret += (
                    str(instance_var_name) + "=\n" + textwrap.indent(value_repr, "  ")
                )
            else:
                ret += str(instance_var_name) + "=" + value_repr
        return ret


class ConditionError(DbCQuantumError):
    """
    The base class for exceptions that are related to assertion failures in DbCQuantum.
    """

    def __init__(self, message: str, info: ConditionErrorInfo):
        super().__init__(message + "\n" + str(info))


@dataclass
class StateConditionErrorInfo(Generic[P], ConditionErrorInfo):
    """
    The dataclass for the information of StateConditionError.
    """

    condition_name: str  #: The name (tag) of the assertion.
    pre_state: Statevector | None  #: The pre-state of the circuit
    post_state: Statevector | None  #: The post-state of the circuit
    param: P  #: runtime parameter for assertions.


class StateConditionError(Generic[P], ConditionError):
    """
    Exceptions for an assertion failure related to the quantum states of a quantum circuit.
    """

    def __init__(self, info: StateConditionErrorInfo[P]):
        super().__init__(f"Condition Error occured in '{info.condition_name}'", info)
        self.info: StateConditionErrorInfo[P] = info  #: information


@dataclass
class SuperpositionInfo:
    """
    The dataclass for the information of superposition states.
    """

    l: list[
        tuple[complex, Statevector]
    ]  #: A list of pairs of a statevector and its amplitude.

    def __repr__(self) -> str:
        ret: list[str] = []
        for a, s in self.l:
            ret.append(str(a) + " * \n" + textwrap.indent(repr(s), "    "))
        return "  " + "\n+ ".join(ret)


@dataclass
class SuperpositionStateConditionErrorInfo(Generic[P], ConditionErrorInfo):
    """
    The dataclass for the information of SuperpositionStateConditionError.
    """

    condition_name: str  #: The name (tag) of the assertion (superposition).
    pre_state: Statevector | None  #: The pre-state of the circuit
    post_state: Statevector | None  #: The post-state of the circuit
    pre_superposition_states: (
        SuperpositionInfo | None
    )  #: Information regarding the superposition states of the pre-state.
    post_superposition_states: (
        SuperpositionInfo | None
    )  #: Information regarding the superposition states of the post-state.
    state1: str  #: The first state.
    state2: str  #: The second state.
    param: P  #: runtime parameter for assertions.


class SuperpositionStateConditionError(Generic[P], ConditionError):
    """
    Exceptions that are raised when there is something wrong with
    the user-specified decomposition of quantum states.
    This error is raised when the expected state and the actual state are inconsistent.
    """

    def __init__(self, info: SuperpositionStateConditionErrorInfo[P]):
        super().__init__(
            f"The {info.state1} and {info.state2} don't match in '{info.condition_name}'",
            info,
        )
        self.info: SuperpositionStateConditionErrorInfo[P] = info  #: information


@dataclass
class MeasureConditionErrorInfo(Generic[R, P], ConditionErrorInfo):
    """
    The dataclass for the information of MeasureConditionError.
    """

    condition_name: str  #: The name (tag) of the assertion.
    pre_measure_state: Statevector  #: The quantum state just before the measurement.
    result: Result  #: The raw data of measurement.
    postprocessed_result: R  #: The postprocessed measurement result.
    param: P  #: runtime parameter for assertions.


class MeasureConditionError(Generic[R, P], ConditionError):
    """
    Exceptions for an assertion failure related to measurement or postprocessing
    the measurement result.
    """

    def __init__(self, info: MeasureConditionErrorInfo[R, P]):
        super().__init__(f"Condition Error occured in '{info.condition_name}'", info)
        self.info: MeasureConditionErrorInfo[R, P] = info
