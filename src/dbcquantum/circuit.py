"""The main module of this framework.
It consists of the AssertQuantumCircuit class and the AQCMeasure class."""

from __future__ import annotations

import copy
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Generic, Iterable, TypeAlias, TypeVar, cast, overload

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library import (
    Barrier,
    CCXGate,
    CPhaseGate,
    CXGate,
    HGate,
    IGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    RZXGate,
    SdgGate,
    SGate,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    UGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.extensions import Initialize
from qiskit.opflow.operator_base import OperatorBase
from qiskit.quantum_info import Statevector
from qiskit.result.result import Result
from qiskit_aer import AerSimulator
from qiskit_aer.backends.aerbackend import AerBackend

from . import _mapping_utils, err, utils
from .common import QubitMapping, QubitSpecifier

P = TypeVar("P", contravariant=True)  # Type for a parameter that is used in conditions.
Q = TypeVar("Q")  # Type for a parameter that is used in conditions.
R = TypeVar("R")  # Type for a final result.

# Generic type for values that can be converted to Statevector
S1 = TypeVar("S1", bound=(Statevector | OperatorBase | list[complex] | np.ndarray))
S2 = TypeVar("S2", bound=(Statevector | OperatorBase | list[complex] | np.ndarray))


# Type for the conditions of a quantum circuit.
# It returns bool from the pre-state and post-state.
CircuitCondition: TypeAlias = Callable[[Statevector, Statevector], bool]

# Type for the post or pre-condition of a quantum circuit.
PrePostCircuitCondition: TypeAlias = Callable[[Statevector], bool]

ParameterValueType: TypeAlias = ParameterExpression | float


class _InstructionSet(Generic[Q], metaclass=ABCMeta):
    """A base class for all InstructionSet: QiskitInstructionSet,
    AQCInstructionSet, AQCSuperpositionInstructionSet.
    InstructionSet is the set of instructions of the whole (we call global) circuit.

    We do not assume that users touch this class or its child classes directly.

    InstructionSet may be nested.
    We refer to the immediately above InstructionSet as "parent".

    Args:
        Generic (Q): Type for a parameter that is used in conditions.
        metaclass: Defaults to ABCMeta.
    """

    def __init__(self):
        pass

    @abstractmethod
    def run(
        self,
        global_pre_state: Statevector,
        global_num_qubits: int,
        parent_global_mapping: QubitMapping,
        param: Q,
    ) -> Statevector:
        """Run instructions of this class. It evolves the whole circuit's state.

        Args:
            global_pre_state (Statevector): The whole circuit's state to be evolved by
                    instructions of this class.
            global_num_qubits (int): The number of qubits of the whole circuit.
            parent_global_mapping (QubitMapping): The qubits' mapping from the whole circuit
                    to the parent InstructionSet.
            param (Q): A parameter that is used in conditions

        Returns:
            Statevector: The evolved state.
        """
        pass

    @abstractmethod
    def inverse(self) -> _InstructionSet:
        """Calculate the inverse of all instructions and sort them in reverse order.

        Returns:
            InstructionSet: Inversed version of this class.
                The instance is built from scratch, so it does not affect the original.
        """
        pass

    @abstractmethod
    def remove_assertions(
        self,
        qc: QuantumCircuit,
        parent_global_mapping: QubitMapping,
    ):
        """Convert to Qiskit's circuit.

        Args:
            qc (QuantumCircuit): Qiskit's QuantumCircuit.
                    All instructions will be added to this argument.
            parent_global_mapping (QubitMapping): The qubits' mapping from the whole circuit
                    to the parent InstructionSet.
        """
        pass

    @abstractmethod
    def copy(self) -> _InstructionSet:
        """Make a deep copy enough to avoid side effects after being appended.

        Returns:
            InstructionSet: A copy of this class.
        """
        pass


class _QiskitInstructionSet(_InstructionSet):
    """A set of qiskit's Instructions."""

    def __init__(self, num_qubits):
        self.num_qubits: int = num_qubits
        self.qiskit_instruction_list: list[tuple[Instruction, list[QubitSpecifier]]] = (
            []
        )

    def copy(self) -> _InstructionSet:
        qiskit_instruction_set = _QiskitInstructionSet(self.num_qubits)
        qiskit_instruction_set.qiskit_instruction_list = copy.copy(
            self.qiskit_instruction_list
        )
        return qiskit_instruction_set

    def append(
        self,
        qiskit_instruction: Instruction,
        qargs: list[QubitSpecifier],
    ):
        self.qiskit_instruction_list.append((qiskit_instruction, qargs))

    def run(
        self,
        global_pre_state: Statevector,
        global_num_qubits: int,
        parent_global_mapping: QubitMapping,
        param: Any,
    ) -> Statevector:
        global_mapping: QubitMapping = parent_global_mapping

        qc: QuantumCircuit = QuantumCircuit(global_num_qubits)

        for instruction, qargs in self.qiskit_instruction_list:
            qc.append(instruction, _mapping_utils.apply_all(global_mapping, qargs))

        global_post_state: Statevector = global_pre_state.evolve(qc)
        return global_post_state

    def inverse(self) -> _InstructionSet:
        qiskit_instruction_set: _QiskitInstructionSet = _QiskitInstructionSet(
            self.num_qubits
        )

        for qiskit_instruction, qargs in reversed(self.qiskit_instruction_list):
            qiskit_instruction_set.append(qiskit_instruction.inverse(), qargs)

        return qiskit_instruction_set

    def remove_assertions(
        self,
        qc: QuantumCircuit,
        parent_global_mapping: QubitMapping,
    ):
        global_mapping: QubitMapping = parent_global_mapping
        for instruction, qargs in self.qiskit_instruction_list:
            qc.append(instruction, _mapping_utils.apply_all(global_mapping, qargs))


class _AQCInstructionSet(Generic[P, Q], _InstructionSet[Q]):
    """An InstructionSet of AssertQuantumCircuit[Q]."""

    def __init__(
        self,
        aqc: AssertQuantumCircuit[Q],
        mapping: QubitMapping,
        param_converter: Callable[[P], Q],
    ):
        self.aqc: AssertQuantumCircuit[Q] = aqc
        self.mapping: QubitMapping = mapping
        self.param_converter: Callable[[P], Q] = param_converter

    def run(
        self,
        global_pre_state: Statevector,
        global_num_qubits: int,
        parent_global_mapping: QubitMapping,
        param: P,
    ) -> Statevector:
        global_mapping: QubitMapping = _mapping_utils.compose(
            parent_global_mapping, self.mapping
        )
        global_post_state: Statevector = self.aqc._run(
            global_pre_state,
            global_num_qubits,
            global_mapping,
            self.param_converter(param),
        )
        return global_post_state

    def copy(self) -> _InstructionSet:
        return copy.copy(self)

    def inverse(self) -> _InstructionSet:
        aqc: AssertQuantumCircuit[Q] = self.aqc._inverse()
        aqc_instruction_set: _AQCInstructionSet[P, Q] = _AQCInstructionSet(
            aqc, self.mapping, self.param_converter
        )
        return aqc_instruction_set

    def remove_assertions(
        self,
        qc: QuantumCircuit,
        parent_global_mapping: QubitMapping,
    ):
        global_mapping: QubitMapping = _mapping_utils.compose(
            parent_global_mapping, self.mapping
        )
        self.aqc._remove_assertions(qc, global_mapping)


class _AQCSuperpositionInstructionSet(Generic[P, Q], _InstructionSet[Q]):
    """An InstructionSet to decompose the state of AssertQuantumCircuit[Q]."""

    def __init__(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        mapping: QubitMapping,
        given_local_pre_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, Statevector]]] | None
        ),
        given_local_post_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, Statevector]]] | None
        ),
        param_converter: Callable[[P], list[Q]] | None,
    ):
        self.aqc: AssertQuantumCircuit[Q] = aqc
        self.mapping: QubitMapping = mapping
        self.name: str = name
        self.given_local_pre_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, Statevector]]] | None
        ) = given_local_pre_states_gen
        self.given_local_post_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, Statevector]]] | None
        ) = given_local_post_states_gen
        self.param_converter: Callable[[P], list[Q]] | None = param_converter

    def copy(self) -> _InstructionSet:
        return copy.copy(self)

    def run(
        self,
        global_pre_state: Statevector,
        global_num_qubits: int,
        parent_global_mapping: QubitMapping,
        param: P,
    ) -> Statevector:
        global_mapping: QubitMapping = _mapping_utils.compose(
            parent_global_mapping, self.mapping
        )

        focus_qubits: list[QubitSpecifier] = _mapping_utils.apply_all(
            global_mapping, list(range(self.aqc.num_qubits))
        )

        local_pre_state: Statevector = utils.partial_state(
            global_pre_state, focus_qubits
        )

        local_other_qubits_state: Statevector = utils._partial_state_traceout(
            global_pre_state, focus_qubits
        )

        if self.given_local_pre_states_gen is None:
            raise err.DbCQuantumError(
                "The state vector cannot be decomposed because "
                f"it is not specified in '{self.name}'."
            )
        given_local_pre_states = self.given_local_pre_states_gen(local_pre_state, param)

        sum_given_local_pre_states: Statevector = utils.state_sum(
            map(lambda a: (a[0], a[1]), given_local_pre_states), self.aqc.num_qubits
        )

        if not utils.eq_state(
            sum_given_local_pre_states,
            local_pre_state,
        ):
            raise err.SuperpositionStateConditionError(
                err.SuperpositionStateConditionErrorInfo(
                    self.name,
                    local_pre_state,
                    None,
                    err.SuperpositionInfo(
                        list(map(lambda a: (a[0], a[1]), given_local_pre_states))
                    ),
                    None,
                    state1="pre_state",
                    state2="pre_superposition_states",
                    param=param,
                )
            )

        local_post_states: list[tuple[complex, Statevector]] = []

        param_converter: Callable[[P], list[Q]] = (
            self.param_converter
            if self.param_converter is not None
            else lambda p: list(
                map(lambda _: cast(Q, p), given_local_pre_states)  # This cast is safe.
            )
        )
        converted_params = param_converter(param)

        if len(given_local_pre_states) != len(converted_params):
            raise err.DbCQuantumError(
                "The length of pre_superposition_states is "
                "inconsistent with param_converter."
            )

        for (a, state), converted_param in zip(
            given_local_pre_states,
            converted_params,
            strict=True,
        ):
            result_state: Statevector = self.aqc._run(
                # run on local state
                state,
                self.aqc.num_qubits,
                list(range(self.aqc.num_qubits)),
                converted_param,
            )

            local_post_states.append((a, result_state))

        local_post_state: Statevector = utils.state_sum(
            local_post_states, self.aqc.num_qubits
        )

        global_post_state: Statevector = utils._merge_statevector(
            local_post_state, local_other_qubits_state, focus_qubits
        )

        if self.given_local_post_states_gen is None:
            return global_post_state

        given_local_post_states = self.given_local_post_states_gen(
            local_post_state, param
        )

        sum_local_given_post_states: Statevector = utils.state_sum(
            (map(lambda a: (a[0], a[1]), given_local_post_states)), self.aqc.num_qubits
        )

        if len(given_local_pre_states) != len(given_local_post_states):
            raise err.DbCQuantumError(
                "The length of pre_superposition_states and "
                "post_superposition_states is inconsistent!"
            )
        for (a, p_state), (b, gp_state) in zip(
            local_post_states,
            given_local_post_states,
            strict=True,
        ):
            if not utils.eq_state(
                p_state,
                gp_state,
            ):
                raise err.SuperpositionStateConditionError(
                    err.SuperpositionStateConditionErrorInfo(
                        self.name,
                        local_pre_state,
                        local_post_state,
                        err.SuperpositionInfo(
                            list(map(lambda a: (a[0], a[1]), given_local_pre_states))
                        ),
                        err.SuperpositionInfo(
                            list(map(lambda a: (a[0], a[1]), given_local_post_states))
                        ),
                        state1="evolved pre_superposition_states",
                        state2="post_superposition_states",
                        param=param,
                    )
                )

        if not utils.eq_state(
            sum_local_given_post_states,
            local_post_state,
        ):
            raise err.SuperpositionStateConditionError(
                err.SuperpositionStateConditionErrorInfo(
                    self.name,
                    local_pre_state,
                    local_post_state,
                    err.SuperpositionInfo(
                        list(map(lambda a: (a[0], a[1]), given_local_pre_states))
                    ),
                    err.SuperpositionInfo(
                        list(map(lambda a: (a[0], a[1]), given_local_post_states))
                    ),
                    state1="post_state",
                    state2="post_superposition_states",
                    param=param,
                )
            )

        global_post_state: Statevector = utils._merge_statevector(
            local_post_state, local_other_qubits_state, focus_qubits
        )
        return global_post_state

    def inverse(self) -> _InstructionSet:
        aqc: AssertQuantumCircuit[Q] = self.aqc._inverse()
        aqc_superposition_instruction_set: _AQCSuperpositionInstructionSet[P, Q] = (
            _AQCSuperpositionInstructionSet(
                self.name,
                aqc,
                self.mapping,
                self.given_local_post_states_gen,
                self.given_local_pre_states_gen,
                self.param_converter,
            )
        )
        return aqc_superposition_instruction_set

    def remove_assertions(
        self,
        qc: QuantumCircuit,
        parent_global_mapping: QubitMapping,
    ):
        global_mapping: QubitMapping = _mapping_utils.compose(
            parent_global_mapping, self.mapping
        )
        self.aqc._remove_assertions(qc, global_mapping)


class AssertQuantumCircuit(Generic[P]):
    """
    A class to construct a quantum circuit with assertions.
    This class is parameterized by P, which stands for the runtime parameter for the assertions.
    """

    def __init__(self, num_qubits: int):
        """
        Args:
            num_qubits (int): The number of qubits in the quantum circuit.
        """
        # fmt: off
        self.num_qubits: int = num_qubits  #: The number of qubits in the quantum circuit.
        # fmt: on
        self._instruction_set_list: list[_InstructionSet] = []
        self._conditions_gen: Callable[[P], list[tuple[str, CircuitCondition]]] = (
            lambda p: []
        )
        self._pre_conditions_gen: Callable[
            [P], list[tuple[str, PrePostCircuitCondition]]
        ] = lambda p: []
        self._post_conditions_gen: Callable[
            [P], list[tuple[str, PrePostCircuitCondition]]
        ] = lambda p: []
        self._eq_circuits_gen: Callable[[P], list[tuple[str, QuantumCircuit]]] = (
            lambda p: []
        )

    def copy(self) -> AssertQuantumCircuit[P]:
        """Creates a copy of the quantum circuit.

        Returns:
            AssertQuantumCircuit[P]: A copy of the existing quantum circuit.
        """
        aqc: AssertQuantumCircuit[P] = AssertQuantumCircuit(self.num_qubits)
        aqc._instruction_set_list = list(
            map(
                lambda instruction_set: instruction_set.copy(),
                self._instruction_set_list,
            )
        )
        aqc._conditions_gen = self._conditions_gen
        aqc._pre_conditions_gen = self._pre_conditions_gen
        aqc._post_conditions_gen = self._post_conditions_gen
        aqc._eq_circuits_gen = self._eq_circuits_gen

        return aqc

    def add_conditions_gen(
        self,
        conditions_gen: Callable[
            [P], list[tuple[str, Callable[[Statevector, Statevector], bool]]]
        ],
    ):
        """Adds a generator that generates assertions from the runtime parameter.
        The assertions here are related to both the prestate and poststate of the circuit.
        Each assertion is a tuple where the first element is a tag (name) of the assertion
        and the second element is the body of the assertion.
        The tag is displayed when an error occurs to identify which assertion is not established.
        The body of the assertion is a function that takes two Statevector instances (representing
        the prestate and poststate of the circuit) and returns a boolean indicating
        whether the assertion is satisfied.
        The assertions are checked after the execution of the circuit.

        Args:
            conditions_gen
                (Callable[[P], list[tuple[str, Callable[[Statevector, Statevector], bool]]]]):
                A function that generates a list of assertions from the runtime parameter.
        """
        self._conditions_gen = utils._lazy_composition(
            self._conditions_gen, conditions_gen
        )

    def add_condition(
        self,
        name: str,
        condition: Callable[[Statevector, Statevector], bool],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Adds an assertion that is related to both the prestate and poststate of the circuit.
        The assertion is checked after the execution of the circuit.

        Args:
            name (str):
                The name (tag) of the assertion.
                This is displayed when an error occurs to identify which assertion is
                not established.
            condition (Callable[[Statevector, Statevector], bool]):
                The body of the assertion. It takes the prestate and poststate of
                the circuit and returns a boolean value indicating
                whether the assertion is satisfied.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are related to the assertions.
                Only the statevector of the specified qubits is extracted and passed
                to the condition.
                If not specified, all qubits are considered. Defaults to None.
        """

        def _condition(
            pre_state: Statevector, post_state: Statevector, param: P
        ) -> bool:
            return condition(pre_state, post_state)

        self.add_condition_use_param(name, _condition, focus_qubits)

    def add_condition_use_param(
        self,
        name: str,
        condition: Callable[[Statevector, Statevector, P], bool],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Adds an assertion that is related to both the prestate and poststate of the circuit.
        The assertion can also use the runtime parameter.
        The assertion is checked after the execution of the circuit.

        Args:
            name (str):
                The name (tag) of the assertion.
                This is displayed when an error occurs to identify which assertion is
                not established.
            condition (Callable[[Statevector, Statevector, P], bool]):
                The body of the assertion.
                It takes the prestate, poststate, and the runtime parameter,
                and returns a boolean value indicating whether the assertion is satisfied.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are related to the assertions.
                Only the statevector of the specified qubits is extracted and passed
                to the condition.
                If not specified, all qubits are considered. Defaults to None.
        """
        if focus_qubits is None:
            focus_qubits = range(self.num_qubits)
        _focus_qubits = list(focus_qubits)

        def _condition(param: P):
            def f(pre_state: Statevector, post_state: Statevector) -> bool:
                return condition(
                    utils.partial_state(pre_state, _focus_qubits),
                    utils.partial_state(post_state, _focus_qubits),
                    param,
                )

            return [(name, f)]

        self.add_conditions_gen(_condition)

    def add_pre_conditions_gen(
        self,
        conditions_gen: Callable[[P], list[tuple[str, Callable[[Statevector], bool]]]],
    ):
        """Adds a generator that generates assertions from the runtime parameter.
        The assertions here are related to only the prestate of the circuit.
        Each assertion is a tuple where the first element is a tag (name) of the assertion
        and the second element is the body of the assertion.
        The tag is displayed when an error occurs to identify which assertion is not established.
        The body of the assertion is a function that takes a Statevector instance (representing
        the prestate of the circuit) and returns a boolean indicating whether the assertion
        is satisfied.
        The assertions are checked before the execution of the circuit.

        Args:
            conditions_gen (Callable[[P], list[tuple[str, Callable[[Statevector], bool]]]]):
                A function that generates a list of assertions from the runtime parameter.
        """
        self._pre_conditions_gen = utils._lazy_composition(
            self._pre_conditions_gen, conditions_gen
        )

    def add_pre_condition(
        self,
        name: str,
        condition: Callable[[Statevector], bool],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Adds an assertion that is related only to the prestate of the circuit.
        The assertion is checked before the execution of the circuit.

        Args:
            name (str):
                The name (tag) of the assertion.
                This is displayed when an error occurs to identify which assertion is
                not established.
            condition (Callable[[Statevector], bool]):
                The body of the assertion. It takes the prestate of
                the circuit and returns a boolean value indicating
                whether the assertion is satisfied.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are related to the assertions.
                Only the statevector of the specified qubits is extracted and passed
                to the condition.
                If not specified, all qubits are considered. Defaults to None.
        """

        def _condition(pre_state: Statevector, param: P) -> bool:
            return condition(pre_state)

        self.add_pre_condition_use_param(name, _condition, focus_qubits)

    def add_pre_condition_use_param(
        self,
        name: str,
        condition: Callable[[Statevector, P], bool],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Adds an assertion that is related only to the prestate of the circuit.
        The assertion can also use the runtime parameter.
        The assertion is checked before the execution of the circuit.

        Args:
            name (str):
                The name (tag) of the assertion.
                This is displayed when an error occurs to identify which assertion is
                not established.
            condition (Callable[[Statevector, P], bool]):
                The body of the assertion. It takes the prestate of the circuit
                and the runtime parameter, and returns a boolean value indicating
                whether the assertion is satisfied.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are related to the assertions.
                Only the statevector of the specified qubits is extracted and passed
                to the condition.
                If not specified, all qubits are considered. Defaults to None.
        """
        if focus_qubits is None:
            focus_qubits = range(self.num_qubits)
        _focus_qubits = list(focus_qubits)

        def _condition(param: P):
            def f(pre_state: Statevector) -> bool:
                return condition(
                    utils.partial_state(pre_state, _focus_qubits),
                    param,
                )

            return [(name, f)]

        self.add_pre_conditions_gen(_condition)

    def add_post_conditions_gen(
        self,
        conditions_gen: Callable[[P], list[tuple[str, Callable[[Statevector], bool]]]],
    ):
        """Adds a generator that generates assertions from the runtime parameter.
        The assertions here are related to only the poststate of the circuit.
        Each assertion is a tuple where the first element is a tag (name) of the assertion
        and the second element is the body of the assertion.
        The tag is displayed when an error occurs to identify which assertion is not established.
        The body of the assertion is a function that takes a Statevector instance (representing
        the poststate of the circuit) and returns a boolean indicating whether the assertion
        is satisfied.
        The assertions are checked after the execution of the circuit,
        or before the execution when inverting the circuit.

        Args:
            conditions_gen (Callable[[P], list[tuple[str, Callable[[Statevector], bool]]]]):
                A function that generates a list of assertions from the runtime parameter.
        """
        self._post_conditions_gen = utils._lazy_composition(
            self._post_conditions_gen, conditions_gen
        )

    def add_post_condition(
        self,
        name: str,
        condition: Callable[[Statevector], bool],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Adds an assertion that is related only to the poststate of the circuit.
        The assertion is checked after the execution of the circuit,
        or before the execution when inverting the circuit.

        Args:
            name (str):
                The name (tag) of the assertion.
                This is displayed when an error occurs to identify which assertion is
                not established.
            condition (Callable[[Statevector], bool]):
                The body of the assertion. It takes the poststate of
                the circuit and returns a boolean value indicating
                whether the assertion is satisfied.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are related to the assertions.
                Only the statevector of the specified qubits is extracted and passed
                to the condition.
                If not specified, all qubits are considered. Defaults to None.
        """

        def _condition(post_state: Statevector, param: P) -> bool:
            return condition(post_state)

        self.add_post_condition_use_param(name, _condition, focus_qubits)

    def add_post_condition_use_param(
        self,
        name: str,
        condition: Callable[[Statevector, P], bool],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Adds an assertion that is related only to the poststate of the circuit.
        The assertion can also use the runtime parameter.
        The assertion is checked after the execution of the circuit,
        or before the execution when inverting the circuit.

        Args:
            name (str):
                The name (tag) of the assertion.
                This is displayed when an error occurs to identify which assertion is
                not established.
            condition (Callable[[Statevector, P], bool]):
                The body of the assertion. It takes the poststate of the circuit
                and the runtime parameter, and returns a boolean value indicating
                whether the assertion is satisfied.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are related to the assertions.
                Only the statevector of the specified qubits is extracted and passed
                to the condition.
                If not specified, all qubits are considered. Defaults to None.
        """
        if focus_qubits is None:
            focus_qubits = range(self.num_qubits)
        _focus_qubits = list(focus_qubits)

        def _condition(param: P):
            def f(post_state: Statevector) -> bool:
                return condition(
                    utils.partial_state(post_state, _focus_qubits),
                    param,
                )

            return [(name, f)]

        self.add_post_conditions_gen(_condition)

    def add_eq_circuits_gen(
        self, circuit_gen: Callable[[P], list[tuple[str, QuantumCircuit]]]
    ):
        """Adds a generator that generates a list of pairs consisting of a tag (name) and a circuit.
        The generated circuits are checked for equivalence with this circuit.
        The tag is displayed when an error occurs to identify which circuit is not equivalent.
        The equivalence is checked by comparing the state vectors after running them.
        The global phases are ignored.

        Args:
            circuit_gen (Callable[[P], list[tuple[str, QuantumCircuit]]]):
                A function that generates a list of pairs,
                each consisting of a tag and a QuantumCircuit, from the runtime parameter.
        """
        self._eq_circuits_gen = utils._lazy_composition(
            self._eq_circuits_gen, circuit_gen
        )

    def add_eq_circuit(self, name: str, circuit: QuantumCircuit):
        """Adds a circuit that is checked for equivalence with this circuit.
        The equivalence is checked by comparing the state vectors after running them.
        The global phases are ignored.

        Args:
            name (str):
                The name (tag) of the circuit.
                This is displayed when an error occurs to identify which circuit is not equivalent.
            circuit (QuantumCircuit):
                A quantum circuit that is checked for equivalence with this circuit.
        """
        self.add_eq_circuits_gen(lambda p: [(name, circuit)])

    def add_eq_circuit_use_param(
        self, name: str, circuit_gen: Callable[[P], QuantumCircuit]
    ):
        """Adds a generator that generates a circuit.
        The generated circuit is checked for equivalence with this circuit.
        The equivalence is checked by comparing the state vectors after running them.
        The global phases are ignored.

        Args:
            name (str):
                The name (tag) of the circuit.
                This is displayed when an error occurs to identify which circuit is not equivalent.
            circuit_gen (Callable[[P], QuantumCircuit]):
                A function that generates a QuantumCircuit from the runtime parameter.
        """

        def _circuit(param: P):
            return [(name, circuit_gen(param))]

        self.add_eq_circuits_gen(_circuit)

    def _append_check(
        self,
        instruction: AssertQuantumCircuit | Instruction,
        qargs: list[QubitSpecifier],
    ):
        if instruction.num_qubits > self.num_qubits:
            raise err.DbCQuantumError("The instruction is bigger than the circuit!")
        if instruction.num_qubits != len(qargs):
            raise err.DbCQuantumError(
                "The length of qargs is inconsistent with the instruction!"
            )
        if any(map(lambda q: q < 0 or q >= self.num_qubits, qargs)):
            raise err.DbCQuantumError("The specified qubits are out of range!")
        if len(qargs) != len(set(qargs)):
            raise err.DbCQuantumError("duplicate qubit arguments")

    @overload
    def append(
        self,
        instruction: AssertQuantumCircuit[P] | Instruction | QuantumCircuit,
        qargs: Iterable[QubitSpecifier],
    ) -> None: ...

    @overload
    def append(
        self,
        instruction: AssertQuantumCircuit[Q],
        qargs: Iterable[QubitSpecifier],
        param_converter: Callable[[P], Q],
    ) -> None: ...

    def append(
        self,
        instruction: AssertQuantumCircuit[Q] | Instruction | QuantumCircuit,
        qargs: Iterable[QubitSpecifier],
        param_converter: Callable[[P], Q] | None = None,
    ) -> None:
        """Appends an instruction to the end of the circuit.

        Args:
            instruction (AssertQuantumCircuit[Q] | Instruction | QuantumCircuit):
                The instruction to be added.
                It can be an AssertQuantumCircuit, a quantum gate, or a quantum circuit.
            qargs (Iterable[QubitSpecifier]):
                The qubits that the instruction ranges over.
            param_converter (Callable[[P], Q] | None, optional):
                A funtion that converts the runtime parameter for this circuit
                to the parameter for the appended component circuit.
                If not provided, it defaults to the identity function.
        """
        _qargs: list[QubitSpecifier] = list(qargs)

        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction(label=utils._decompose_label)
        self._append_check(instruction, _qargs)

        if param_converter is None:
            param_converter = lambda p: cast(Q, p)  # This cast is safe.

        match instruction:
            case AssertQuantumCircuit():
                self._append_assert_quantum_circuit(
                    instruction, _qargs, param_converter
                )
            case _:
                self._append_qiskit_instruction(instruction, _qargs)

    def _append_qiskit_instruction(
        self,
        qiskit_instruction: Instruction,
        qargs: list[QubitSpecifier],
    ):
        if len(self._instruction_set_list) != 0 and isinstance(
            self._instruction_set_list[-1], _QiskitInstructionSet
        ):
            self._instruction_set_list[-1].append(qiskit_instruction, qargs)
        else:
            qiskit_instruction_set: _QiskitInstructionSet = _QiskitInstructionSet(
                self.num_qubits
            )
            qiskit_instruction_set.append(qiskit_instruction, qargs)
            self._instruction_set_list.append(qiskit_instruction_set)

    def _append_assert_quantum_circuit(
        self,
        aqc: AssertQuantumCircuit[Q],
        qargs: list[QubitSpecifier],
        param_converter: Callable[[P], Q],
    ):
        aqc_instruction: _AQCInstructionSet[P, Q] = _AQCInstructionSet(
            aqc, qargs, param_converter
        )
        self._instruction_set_list.append(aqc_instruction)

    @overload
    def append_superposition_basis_from_param(
        self,
        name: str,
        aqc: AssertQuantumCircuit[P],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_state_basis: Callable[[P], list[S1]] | None = None,
        post_state_basis: Callable[[P], list[S2]] | None = None,
        param_converter: Callable[[P], list[P]] | None = None,
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ): ...

    @overload
    def append_superposition_basis_from_param(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_state_basis: Callable[[P], list[S1]] | None = None,
        post_state_basis: Callable[[P], list[S2]] | None = None,
        param_converter: Callable[[P], list[Q]],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ): ...

    def append_superposition_basis_from_param(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_state_basis: Callable[[P], list[S1]] | None = None,
        post_state_basis: Callable[[P], list[S2]] | None = None,
        param_converter: Callable[[P], list[Q]] | None = None,
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Appends an AssertQuantumCircuit to the end of the quantum circuit.
        This method is designed to use quantum circuits in a superposition state,
        enabling the reuse of assertions that are typically valid only for non-superposition states.
        The input state is decomposed according to a user-specified method,
        and the circuit is executed for each decomposed state separately.
        This approach allows for the reuse of assertions within the circuit.
        The final state is derived by combining all the results from the decomposed states.

        Args:
            name (str):
                The name (tag) associated with the decomposition.
                This tag is displayed when an error occurs during the decomposition
                process using the user-specified method.
            aqc (AssertQuantumCircuit[Q]):
                The instruction to be added.
            qargs (Iterable[QubitSpecifier]):
                The qubits that the instruction ranges over.
            pre_state_basis (Callable[[P], list[S1]] | None, optional):
                A function that generates the list of decomposed pre-states from the runtime
                parameter.
                The elements of this list should be linearly independent.
                The amplitudes will be calculated automatically.
                If not provided, the decomposition will be skipped.
            post_state_basis (Callable[[P], list[S2]] | None, optional):
                A function that generates the list of decomposed post-states from the runtime
                parameter.
                This list is used to verify whether each post state is as expected.
                The elements of this list should be linearly independent.
                The amplitudes will be calculated automatically.
                If not provided, the verification will be skipped.
            param_converter (Callable[[P], list[Q]] | None, optional):
                A function that converts the runtime parameter for this circuit
                into a parameter for the appended component circuit.
                The elements of the list will be distributed to each decomposed state.
                If not provided, it defaults to the identity function.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are in superposition states.
                If not specified, all qubits are considered. Defaults to None.
        """
        # TODO: Error handling
        _qargs = list(qargs)

        if focus_qubits is None:
            _focus_qubits = list(range(aqc.num_qubits))
        else:
            _focus_qubits = list(focus_qubits)

        pre_superposition_gen = None
        if pre_state_basis is not None:
            # if len(_qargs) != _pre_basis[0].num_qubits:
            #     raise RuntimeError(
            #         "The size of pre_state_basis is inconsistent with qargs."
            #     )

            def _pre_superposition_gen(
                s: Statevector, param: P
            ) -> list[tuple[complex, Statevector]]:
                _pre_basis = [utils.to_Statevector(b) for b in pre_state_basis(param)]
                local_other_s = utils._partial_state_traceout(s, _focus_qubits)
                _pre_basis_global = [
                    utils._merge_statevector(basis, local_other_s, _focus_qubits)
                    for basis in _pre_basis
                ]
                coeff = utils.coeff_of_basis(s, _pre_basis_global)
                return list(zip(coeff, _pre_basis_global))

            pre_superposition_gen = _pre_superposition_gen

        post_superposition_gen = None
        if post_state_basis is not None:
            # if len(_qargs) != _post_basis[0].num_qubits:
            #     raise RuntimeError(
            #         "The size of post_state_basis is inconsistent with qargs."
            #     )

            def _post_superposition_gen(
                s: Statevector, param: P
            ) -> list[tuple[complex, Statevector]]:
                _post_basis = [utils.to_Statevector(b) for b in post_state_basis(param)]
                local_other_s = utils._partial_state_traceout(s, _focus_qubits)
                _post_basis_global = [
                    utils._merge_statevector(basis, local_other_s, _focus_qubits)
                    for basis in _post_basis
                ]
                coeff = utils.coeff_of_basis(s, _post_basis_global)
                return list(zip(coeff, _post_basis_global))

            post_superposition_gen = _post_superposition_gen

        self._append_superposition_gen(
            name,
            aqc,
            pre_superposition_gen,
            post_superposition_gen,
            _qargs,
            param_converter,
        )

    @overload
    def append_superposition_basis(
        self,
        name: str,
        aqc: AssertQuantumCircuit[P],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_state_basis: Iterable[S1] | None = None,
        post_state_basis: Iterable[S2] | None = None,
        param_converter: Callable[[P], list[P]] | None = None,
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ): ...

    @overload
    def append_superposition_basis(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_state_basis: Iterable[S1] | None = None,
        post_state_basis: Iterable[S2] | None = None,
        param_converter: Callable[[P], list[Q]],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ): ...

    def append_superposition_basis(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_state_basis: Iterable[S1] | None = None,
        post_state_basis: Iterable[S2] | None = None,
        param_converter: Callable[[P], list[Q]] | None = None,
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Appends an AssertQuantumCircuit to the end of the quantum circuit.
        This method is designed to use quantum circuits in a superposition state,
        enabling the reuse of assertions that are typically valid only for non-superposition states.
        The input state is decomposed according to a user-specified method,
        and the circuit is executed for each decomposed state separately.
        This approach allows for the reuse of assertions within the circuit.
        The final state is derived by combining all the results from the decomposed states.

        Args:
            name (str):
                The name (tag) associated with the decomposition.
                This tag is displayed when an error occurs during the decomposition
                process using the user-specified method.
            aqc (AssertQuantumCircuit[Q]):
                The instruction to be added.
            qargs (Iterable[QubitSpecifier]):
                The qubits that the instruction ranges over.
            pre_state_basis (Iterable[S1] | None, optional):
                A list of decomposed pre-states.
                The elements of this list should be linearly independent.
                The amplitudes will be calculated automatically.
                If not provided, the decomposition will be skipped.
            post_state_basis (Iterable[S2] | None, optional):
                A list of decomposed post-states.
                This list is used to verify whether each post state is as expected.
                The elements of this list should be linearly independent.
                The amplitudes will be calculated automatically.
                If not provided, the verification will be skipped.
            param_converter (Callable[[P], list[Q]] | None, optional):
                A function that converts the runtime parameter for this circuit
                into a parameter for the appended component circuit.
                The elements of the list will be distributed to each decomposed state.
                If not provided, it defaults to the identity function.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are in superposition states.
                If not specified, all qubits are considered. Defaults to None.
        """
        if pre_state_basis is None:
            _pre_state_basis = None
        else:
            pre_state_basis = list(pre_state_basis)
            _pre_state_basis = lambda _: pre_state_basis

        if post_state_basis is None:
            _post_state_basis = None
        else:
            post_state_basis = list(post_state_basis)
            _post_state_basis = lambda _: post_state_basis

        self.append_superposition_basis_from_param(
            name,
            aqc,
            qargs,
            pre_state_basis=_pre_state_basis,
            post_state_basis=_post_state_basis,
            param_converter=param_converter,
            focus_qubits=focus_qubits,
        )

    @overload
    def append_superposition(
        self,
        name: str,
        aqc: AssertQuantumCircuit[P],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_superposition_states: Iterable[tuple[complex, S1]] | None = None,
        post_superposition_states: Iterable[tuple[complex, S2]] | None = None,
        param_converter: Callable[[P], list[P]] | None = None,
    ): ...

    @overload
    def append_superposition(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_superposition_states: Iterable[tuple[complex, S1]] | None = None,
        post_superposition_states: Iterable[tuple[complex, S2]] | None = None,
        param_converter: Callable[[P], list[Q]],
    ): ...

    def append_superposition(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_superposition_states: Iterable[tuple[complex, S1]] | None = None,
        post_superposition_states: Iterable[tuple[complex, S2]] | None = None,
        param_converter: Callable[[P], list[Q]] | None = None,
    ):
        """Appends an AssertQuantumCircuit to the end of the quantum circuit.
        This method is designed to use quantum circuits in a superposition state,
        enabling the reuse of assertions that are typically valid only for non-superposition states.
        The input state is decomposed according to a user-specified method,
        and the circuit is executed for each decomposed state separately.
        This approach allows for the reuse of assertions within the circuit.
        The final state is derived by combining all the results from the decomposed states.

        Args:
            name (str):
                The name (tag) associated with the decomposition.
                This tag is displayed when an error occurs during the decomposition
                process using the user-specified method.
            aqc (AssertQuantumCircuit[Q]):
                The instruction to be added.
            qargs (Iterable[QubitSpecifier]):
                The qubits that the instruction ranges over.
            pre_superposition_states (Iterable[tuple[complex, S1]] | None, optional):
                A list of decomposed pre-states along with their amplitudes.
                If not provided, the decomposition will be skipped.
            post_superposition_states (Iterable[tuple[complex, S2]] | None, optional):
                A list of decomposed post-states along with their amplitudes.
                This list is used to verify whether each post state is as expected.
                If not provided, the verification will be skipped.
            param_converter (Callable[[P], list[Q]] | None, optional):
                A function that converts the runtime parameter for this circuit
                into a parameter for the appended component circuit.
                The elements of the list will be distributed to each decomposed state.
                If not provided, it defaults to the identity function.
        """
        _pre_superposition_states: list[tuple[complex, S1]] | None = (
            list(pre_superposition_states)
            if pre_superposition_states is not None
            else None
        )

        pre_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S1]]] | None
        ) = (
            (
                lambda s, p: (
                    list(map(lambda a_s: (a_s[0], a_s[1]), _pre_superposition_states))
                )
            )
            if _pre_superposition_states is not None
            else None
        )

        _post_superposition_states: list[tuple[complex, S2]] | None = (
            list(post_superposition_states)
            if post_superposition_states is not None
            else None
        )

        post_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S2]]] | None
        ) = (
            (
                lambda s, p: (
                    list(map(lambda a_s: (a_s[0], a_s[1]), _post_superposition_states))
                )
            )
            if _post_superposition_states is not None
            else None
        )

        self._append_superposition_gen(
            name,
            aqc,
            pre_superposition_states_gen,
            post_superposition_states_gen,
            qargs,
            param_converter,
        )

    @overload
    def append_superposition_gen(
        self,
        name: str,
        aqc: AssertQuantumCircuit[P],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S1]]] | None
        ) = None,
        post_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S2]]] | None
        ) = None,
        param_converter: Callable[[P], list[P]] | None = None,
    ): ...

    @overload
    def append_superposition_gen(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S1]]] | None
        ) = None,
        post_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S2]]] | None
        ) = None,
        param_converter: Callable[[P], list[Q]],
    ): ...

    def append_superposition_gen(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        qargs: Iterable[QubitSpecifier],
        *,
        pre_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S1]]] | None
        ) = None,
        post_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S2]]] | None
        ) = None,
        param_converter: Callable[[P], list[Q]] | None = None,
    ):
        """Appends an AssertQuantumCircuit to the end of the quantum circuit.
        This method is designed to use quantum circuits in a superposition state,
        enabling the reuse of assertions that are typically valid only for non-superposition states.
        The input state is decomposed according to a user-specified method,
        and the circuit is executed for each decomposed state separately.
        This approach allows for the reuse of assertions within the circuit.
        The final state is derived by combining all the results from the decomposed states.

        Args:
            name (str):
                The name (tag) associated with the decomposition.
                This tag is displayed when an error occurs during the decomposition
                process using the user-specified method.
            aqc (AssertQuantumCircuit[Q]):
                The instruction to be added.
            qargs (Iterable[QubitSpecifier]):
                The qubits that the instruction ranges over.
            pre_superposition_states_gen
                (Callable[[Statevector, P], list[tuple[complex, S1]]]  |  None, optional):
                A function that generates a list of decomposed pre-states along with their
                amplitudes from the pre-state and runtime parameter.
                If not provided, it defaults to "lambda s p: [(1, s)]".
            post_superposition_states_gen
                (Callable[[Statevector, P], list[tuple[complex, S2]]]  |  None, optional):
                A function that generates a list of decomposed post-states along with their
                amplitudes from the post-state and runtime parameter.
                This function is used to verify whether each post state is as expected.
                If not provided, the verification will be skipped.
            param_converter (Callable[[P], list[Q]] | None, optional):
                A function that converts the runtime parameter for this circuit
                into a parameter for the appended component circuit.
                The elements of the list will be distributed to each decomposed state.
                If not provided, it defaults to the identity function.
        """
        self._append_superposition_gen(
            name,
            aqc,
            pre_superposition_states_gen,
            post_superposition_states_gen,
            qargs,
            param_converter,
        )

    def _append_superposition_gen(
        self,
        name: str,
        aqc: AssertQuantumCircuit[Q],
        pre_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S1]]] | None
        ),
        post_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, S2]]] | None
        ),
        qargs: Iterable[QubitSpecifier],
        param_converter: Callable[[P], list[Q]] | None,
    ):
        _qargs = list(qargs)
        self._append_check(aqc, _qargs)

        _pre_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, Statevector]]] | None
        ) = (
            utils._lazy_map2(
                lambda a_vec_q: (
                    a_vec_q[0],
                    utils.to_Statevector(a_vec_q[1]),
                ),
                pre_superposition_states_gen,
            )
            if pre_superposition_states_gen is not None
            else None
        )

        _post_superposition_states_gen: (
            Callable[[Statevector, P], list[tuple[complex, Statevector]]] | None
        ) = (
            utils._lazy_map2(
                lambda a_vec_q: (
                    a_vec_q[0],
                    utils.to_Statevector(a_vec_q[1]),
                ),
                post_superposition_states_gen,
            )
            if post_superposition_states_gen is not None
            else None
        )

        aqc_superposition_instruction: _AQCSuperpositionInstructionSet = (
            _AQCSuperpositionInstructionSet(
                name,
                aqc.copy(),
                _qargs,
                _pre_superposition_states_gen,
                _post_superposition_states_gen,
                param_converter,
            )
        )

        self._instruction_set_list.append(aqc_superposition_instruction)

    @overload
    def run(
        self: AssertQuantumCircuit[None],
        init_state: (
            Statevector | OperatorBase | list[complex] | np.ndarray | None
        ) = None,
    ) -> Statevector: ...

    @overload
    def run(
        self,
        init_state: (
            Statevector | OperatorBase | list[complex] | np.ndarray | None
        ) = None,
        *,
        param: P,
    ) -> Statevector: ...

    def run(
        self,
        init_state: (
            Statevector | OperatorBase | list[complex] | np.ndarray | None
        ) = None,
        *,
        param: P | None = None,
    ) -> Statevector:
        """Executes the quantum circuit while checking assertions.

        Args:
            init_state
                (Statevector  |  OperatorBase  |  list[complex]  |  np.ndarray  |  None, optional):
                The initial state to run the circuit.
                If omitted, it is set to 0 ... 0.
            param (P | None, optional):
                The runtime parameter for the assertions.

        Returns:
            Statevector:
                The final state after running the circuit.
        """
        if init_state is None:
            init_state = utils.make_zeros_state(self.num_qubits)
        else:
            init_state = utils.to_Statevector(init_state)

        return self._run(
            init_state,
            self.num_qubits,
            list(range(self.num_qubits)),
            cast(P, param),  # This cast is safe. See the above @overload
        )

    def run_without_assertion(
        self,
        init_state: (
            Statevector | OperatorBase | list[complex] | np.ndarray | None
        ) = None,
    ) -> Statevector:
        """Executes the quantum circuit without checking the assertions.

        Args:
            init_state
                (Statevector  |  OperatorBase  |  list[complex]  |  np.ndarray  |  None, optional):
                The initial state to run the circuit.
                If omitted, it is set to 0 ... 0.

        Returns:
            Statevector:
                The final state after running the circuit.
        """
        qc = self.remove_assertions()
        aqc: AssertQuantumCircuit[None] = AssertQuantumCircuit(self.num_qubits)
        aqc.append(qc.to_instruction(), range(self.num_qubits))

        return aqc.run(init_state=init_state)

    def _run(
        self,
        global_pre_state: Statevector,
        global_num_qubits: int,
        global_mapping: QubitMapping,
        param: P,
    ) -> Statevector:
        qubits: list[QubitSpecifier] = _mapping_utils.apply_all(
            global_mapping, list(range(self.num_qubits))
        )
        global_state: Statevector = global_pre_state

        pre_conditions = self._pre_conditions_gen(param)
        if len(pre_conditions) > 0:
            local_pre_state: Statevector = utils.partial_state(global_pre_state, qubits)

            for name, condition in pre_conditions:
                if not (condition(local_pre_state)):
                    raise err.StateConditionError(
                        err.StateConditionErrorInfo(
                            name,
                            local_pre_state,
                            post_state=None,
                            param=param,
                        )
                    )

        for instruction_set in self._instruction_set_list:
            global_state = instruction_set.run(
                global_state, global_num_qubits, global_mapping, param
            )

        global_post_state: Statevector = global_state

        post_conditions = self._post_conditions_gen(param)
        if len(post_conditions) > 0:
            local_post_state: Statevector = utils.partial_state(
                global_post_state, qubits
            )

            for name, condition in post_conditions:
                if not (condition(local_post_state)):
                    raise err.StateConditionError(
                        err.StateConditionErrorInfo(name, None, local_post_state, param)
                    )

        conditions = self._conditions_gen(param)
        if len(conditions) > 0:
            local_pre_state: Statevector = utils.partial_state(global_pre_state, qubits)
            local_post_state: Statevector = utils.partial_state(
                global_post_state, qubits
            )

            for name, condition in conditions:
                if not (condition(local_pre_state, local_post_state)):
                    raise err.StateConditionError(
                        err.StateConditionErrorInfo(
                            name, local_pre_state, local_post_state, param
                        )
                    )

        for name, circuit in self._eq_circuits_gen(param):
            if not (
                utils.eq_state(global_pre_state.evolve(circuit), global_post_state)
            ):
                raise err.StateConditionError(
                    err.StateConditionErrorInfo(
                        name, global_pre_state, global_post_state, param
                    )
                )

        return global_post_state

    def inverse(self) -> AssertQuantumCircuit[P]:
        """Calculates the inverted quantum circuit.
        It is obtained by reversing the order of the instructions,
        inverting quantum gates, and inverting the assertions.
        The pre and post conditions are swapped.
        In addition, the pre and post states will be swapped
        when passing them to the conditions that are related to both pre and post conditions.
        This method inverts a copy of the circuit, so it does not affect the original circuit.

        Returns:
            AssertQuantumCircuit[P]:
                The inverted quantum circuit.
        """
        return self.copy()._inverse()

    def _inverse(self) -> AssertQuantumCircuit[P]:
        aqc: AssertQuantumCircuit[P] = AssertQuantumCircuit(self.num_qubits)

        for instruction_set in reversed(self._instruction_set_list):
            aqc._instruction_set_list.append(instruction_set.inverse())

        aqc._post_conditions_gen = self._pre_conditions_gen
        aqc._pre_conditions_gen = self._post_conditions_gen

        def swap_args(condition: CircuitCondition) -> CircuitCondition:
            def f(
                global_pre_state: Statevector, global_post_state: Statevector
            ) -> bool:
                return condition(global_post_state, global_pre_state)

            return f

        aqc._conditions_gen = utils._lazy_map(
            lambda name_condition: (
                name_condition[0],
                swap_args(name_condition[1]),
            ),
            self._conditions_gen,
        )

        aqc._eq_circuits_gen = utils._lazy_map(
            lambda name_circuit: (name_circuit[0], name_circuit[1].inverse()),
            self._eq_circuits_gen,
        )

        return aqc

    def remove_assertions(self) -> QuantumCircuit:
        """Removes all assertions and returns a QuantumCircuit that is written in Qiskit.
        You can run it on an actual quantum computer by Qiskit
        without providing the runtime parameters for the assertions.
        This method removes assertions from a copy of the circuit,
        so it does not affect the original circuit.

        Returns:
            QuantumCircuit: The assertion-less quantum circuit.
        """
        qc: QuantumCircuit = QuantumCircuit(self.num_qubits)
        self._remove_assertions(qc, list(range(self.num_qubits)))
        return qc.decompose(utils._decompose_label)

    def _remove_assertions(self, qc: QuantumCircuit, global_mapping: QubitMapping):
        for instruction_set in self._instruction_set_list:
            instruction_set.remove_assertions(qc, global_mapping)

    def barrier(self, qargs: QubitSpecifier | Iterable[QubitSpecifier] | None = None):
        """Appends a barrier to the quantum circuit for separating it into parts.

        Args:
            qargs (QubitSpecifier | Iterable[QubitSpecifier] | None, optional):
                The qubits that the instruction ranges over.
                If omitted, it is set to all the qubits.
        """
        if qargs is None:
            qargs = range(self.num_qubits)
        elif isinstance(qargs, int):
            qargs = [int(qargs)]

        qargs_ = list(qargs)
        self.append(Barrier(num_qubits=len(qargs_)), qargs_)

    def ccx(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        target_qubit: QubitSpecifier,
        ctrl_state: str | int | None = None,
    ):
        """
        Appends a CCX gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            control_qubit1 (QubitSpecifier): The first control qubit.
            control_qubit2 (QubitSpecifier): The second control qubit.
            target_qubit (QubitSpecifier): The target qubit.
            ctrl_state (str | int | None, optional): The control state. Defaults to None.
        """
        self.append(
            CCXGate(ctrl_state=ctrl_state),
            [control_qubit1, control_qubit2, target_qubit],
        )

    def toffoli(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        target_qubit: QubitSpecifier,
    ):
        """Appends a toffoli gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            control_qubit1 (QubitSpecifier): The first control qubit.
            control_qubit2 (QubitSpecifier): The second control qubit.
            target_qubit (QubitSpecifier): The target qubit.
        """
        self.ccx(control_qubit1, control_qubit2, target_qubit)

    def cx(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        ctrl_state: str | int | None = None,
    ):
        """Appends a CX gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            control_qubit (QubitSpecifier): The control qubit.
            target_qubit (QubitSpecifier): The target qubit.
            ctrl_state (str | int | None, optional): The control state. Defaults to None.
        """
        self.append(CXGate(ctrl_state=ctrl_state), [control_qubit, target_qubit])

    def h(self, qubit: QubitSpecifier):
        """Appends a H (Hadamard) gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(HGate(), [qubit])

    def i(self, qubit: QubitSpecifier):
        """Appends an I (Identity) gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(IGate(), [qubit])

    def rx(self, theta: ParameterValueType, qubit: QubitSpecifier):
        """Appends an RX gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            theta (ParameterValueType): The rotation angle.
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(RXGate(theta), [qubit])

    def ry(self, theta: ParameterValueType, qubit: QubitSpecifier):
        """Appends an RY gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            theta (ParameterValueType): The rotation angle.
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(RYGate(theta), [qubit])

    def rz(self, phi: ParameterValueType, qubit: QubitSpecifier):
        """Appends an RX gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            phi (ParameterValueType): The rotation angle.
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(RZGate(phi), [qubit])

    def rzx(
        self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier
    ):
        """Appends an RZX gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            theta (ParameterValueType): The rotation angle.
            qubit1 (QubitSpecifier): The first qubit to which the gate will be applied.
            qubit2 (QubitSpecifier): The second qubit to which the gate will be applied.
        """
        self.append(RZXGate(theta), [qubit1, qubit2])

    def s(self, qubit: QubitSpecifier):
        """Appends a S gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(SGate(), [qubit])

    def sdg(self, qubit: QubitSpecifier):
        """Appends a Sdg gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(SdgGate(), [qubit])

    def swap(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier):
        """Appends a Swap gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit1 (QubitSpecifier): The first qubit to which the gate will be applied.
            qubit2 (QubitSpecifier): The second qubit to which the gate will be applied.
        """
        self.append(SwapGate(), [qubit1, qubit2])

    def sxdg(self, qubit: QubitSpecifier):
        """Appends a SXdg gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(SXdgGate(), [qubit])

    def sx(self, qubit: QubitSpecifier):
        """Appends a SX gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(SXGate(), [qubit])

    def tdg(self, qubit: QubitSpecifier):
        """Appends a Tdg gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(TdgGate(), [qubit])

    def t(self, qubit: QubitSpecifier):
        """Appends a T gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(TGate(), [qubit])

    def x(self, qubit: QubitSpecifier):
        """Appends a X gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(XGate(), [qubit])

    def y(self, qubit: QubitSpecifier):
        """Appends a Y gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(YGate(), [qubit])

    def z(self, qubit: QubitSpecifier):
        """Appends a Z gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(ZGate(), [qubit])

    def p(self, theta: ParameterValueType, qubit: QubitSpecifier):
        """Appends a P (Phase) gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            theta (ParameterValueType): The phase angle.
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(PhaseGate(theta), [qubit])

    def cp(
        self,
        theta: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        ctrl_state: str | int | None = None,
    ):
        """Appends a CP (Controlled-Phase) gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            theta (ParameterValueType): The phase angle.
            control_qubit (QubitSpecifier): The control qubit.
            target_qubit (QubitSpecifier): The target qubit.
            ctrl_state (str | int | None, optional): The control state. Defaults to None.
        """
        self.append(
            CPhaseGate(theta, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
        )

    def u(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        qubit: QubitSpecifier,
    ):
        """Appends a U gate to the circuit.
        For more details, refer to the Qiskit documentation:
        https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit

        Args:
            theta (ParameterValueType): The theta parameter for the U gate.
            phi (ParameterValueType): The phi parameter for the U gate.
            lam (ParameterValueType): The lambda parameter for the U gate.
            qubit (QubitSpecifier): The qubit to which the gate will be applied.
        """
        self.append(UGate(theta, phi, lam), [qubit])


class AQCMeasure(Generic[R, P]):
    """
    A class to measure the quantum circuit and postprocess the measurement result.
    This class is parameterized by R and P,
    which stand for the postprocessed measurement result and the runtime parameter
    for the assertions respectively.
    """

    @overload
    def __init__(
        self,
        aqc: AssertQuantumCircuit[P],
        postprocess: Callable[[Result], R],
        qubit: Iterable[QubitSpecifier] | None = None,
    ): ...

    @overload
    def __init__(
        self,
        aqc: AssertQuantumCircuit[Q],
        postprocess: Callable[[Result], R],
        qubit: Iterable[QubitSpecifier] | None = None,
        *,
        param_converter: Callable[[P], Q],
    ): ...

    def __init__(
        self,
        aqc: AssertQuantumCircuit[Q],
        postprocess: Callable[[Result], R],
        qubit: Iterable[QubitSpecifier] | None = None,
        *,
        param_converter: Callable[[P], Q] | None = None,
    ):
        """
        Args:
            aqc (AssertQuantumCircuit[Q]):
                A quantum circuit to be measured.
            postprocess (Callable[[Result], R]):
                Postprocessing of the measurement result.
            qubit (Iterable[QubitSpecifier] | None, optional):
                The qubits to be measured.
                If omitted, all qubits will be measured.
            param_converter (Callable[[P], Q] | None, optional):
                A function that converts the runtime parameter for this class
                to the parameter for the inner quantum circuit.
                If not provided, it defaults to the identity function.
        """
        _qubit: list[QubitSpecifier]
        if qubit is None:
            _qubit = list(range(aqc.num_qubits))
        else:
            _qubit = list(qubit)

        if param_converter is None:
            param_converter = lambda p: cast(Q, p)  # This cast is safe.

        self.num_qubits: int = (
            aqc.num_qubits
        )  #: The number of qubits in the quantum circuit.
        self._conditions_gen: Callable[
            [P], list[tuple[str, Callable[[Statevector, Result, R], bool]]]
        ] = lambda p: []
        self._aqc: AssertQuantumCircuit[Q] = aqc.copy()
        self._qubit: list[QubitSpecifier] = _qubit
        self._cbit: list[int] = list(range(len(self._qubit)))
        self._postprocess: Callable[[Result], R] = postprocess
        self._param_converter: Callable[[P], Q] = param_converter

    def add_conditions_gen(
        self,
        conditon_gen: Callable[
            [P], list[tuple[str, Callable[[Statevector, Result, R], bool]]]
        ],
    ):
        """Adds a generator that generates assertions from the runtime parameter.
        The assertions here are related to the quantum state just before the measurement,
        the raw data of the measurement result, and the postprocessed measurement result.
        Each assertion is a tuple where the first element is a tag (name) of the assertion
        and the second element is the body of the assertion.
        The tag is displayed when an error occurs to identify which assertion is not established.
        The body of the assertion is a function that takes the quantum state just before
        the measurement, the raw data of the measurement result, and the postprocessed
        measurement result, and returns a boolean indicating whether the assertion is satisfied.
        The assertions are checked after the execution of the circuit.

        Args:
            conditon_gen
                (Callable[ [P], list[tuple[str, Callable[[Statevector, Result, R], bool]]] ]):
                A function that generates a list of assertions from the runtime parameter.
        """
        self._conditions_gen = utils._lazy_composition(
            self._conditions_gen, conditon_gen
        )

    def add_condition(
        self,
        name: str,
        condition: Callable[[Statevector, Result, R], bool],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Adds an assertion that is related to the quantum state just before the measurement,
        the raw data of the measurement result, and the postprocessed measurement result.

        Args:
            name (str):
                The name (tag) of the assertion.
                This is displayed when an error occurs to identify which assertion is
                not established.
            condition (Callable[[Statevector, Result, R], bool]):
                The body of the assertion. It takes the quantum state just before
                the measurement, the raw data of the measurement result,
                and the postprocessed measurement result,
                and returns a boolean indicating whether the assertion is satisfied.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are related to the assertions.
                Only the statevector of the specified qubits is extracted and passed
                to the condition.
                This parameter only affects the statevector passed to the assertion.
                If not specified, all qubits are considered.
        """

        def _condition(
            pre_state: Statevector, result: Result, postprocessed_result: R, param: P
        ) -> bool:
            return condition(pre_state, result, postprocessed_result)

        self.add_condition_use_param(name, _condition, focus_qubits)

    def add_condition_use_param(
        self,
        name: str,
        condition: Callable[[Statevector, Result, R, P], bool],
        focus_qubits: Iterable[QubitSpecifier] | None = None,
    ):
        """Adds an assertion that is related to the quantum state just before the measurement,
        the raw data of the measurement result, and the postprocessed measurement result.
        The assertion can also use the runtime parameter.

        Args:
            name (str):
                The name (tag) of the assertion.
                This is displayed when an error occurs to identify which assertion is
                not established.
            condition (Callable[[Statevector, Result, R, P], bool]):
                The body of the assertion. It takes the quantum state just before
                the measurement, the raw data of the measurement result,
                the postprocessed measurement result and rutime parameter,
                and returns a boolean indicating whether the assertion is satisfied.
            focus_qubits (Iterable[QubitSpecifier] | None, optional):
                The qubits that are related to the assertions.
                Only the statevector of the specified qubits is extracted and passed
                to the condition.
                This parameter only affects the statevector passed to the assertion.
                If not specified, all qubits are considered.
        """
        if focus_qubits is None:
            focus_qubits = range(self.num_qubits)
        _focus_qubits = list(focus_qubits)

        def _condition(param: P):
            def f(
                pre_measure_state: Statevector, result: Result, postprocessed_result: R
            ) -> bool:
                return condition(
                    utils.partial_state(pre_measure_state, _focus_qubits),
                    result,
                    postprocessed_result,
                    param,
                )

            return [(name, f)]

        self.add_conditions_gen(_condition)

    @overload
    def run(
        self: AQCMeasure[R, None],
        shots: int,
        *,
        backend: AerBackend = AerSimulator(),
        init_state: (
            Statevector | OperatorBase | list[complex] | np.ndarray | None
        ) = None,
        seed_simulator: None | int = None,
    ) -> R: ...

    @overload
    def run(
        self,
        shots: int,
        *,
        backend: AerBackend = AerSimulator(),
        init_state: (
            Statevector | OperatorBase | list[complex] | np.ndarray | None
        ) = None,
        param: P,
        seed_simulator: None | int = None,
    ) -> R: ...

    def run(
        self,
        shots: int,
        *,
        backend: AerBackend = AerSimulator(),
        init_state: (
            Statevector | OperatorBase | list[complex] | np.ndarray | None
        ) = None,
        param: P | None = None,
        seed_simulator: None | int = None,
    ) -> R:
        """Executes the quantum circuit, measures qubits,
        and postprocesses the measurement result while checking assertions.

        Args:
            shots (int):
                The number of repetitions of the circuit to be run.
            backend (AerBackend, optional):
                The backend to execute the circuit on. Defaults to AerSimulator().
            init_state
                (Statevector  |  OperatorBase  |  list[complex]  |  np.ndarray  |  None, optional):
                The initial state to run the circuit.
                If omitted, it is set to 0 ... 0.
            param (P | None, optional):
                The runtime parameter for the assertions.
                If omitted, it is set to None.
            seed_simulator (None | int, optional):
                The seed of the simulator to fix the measurement result on each run.

        Returns:
            R: The postprocessed measurement result.
        """
        param = cast(P, param)  # This cast is safe. Please see the above @overload
        pre_measure_state: Statevector = self._aqc.run(
            init_state=init_state, param=self._param_converter(param)
        )
        qc: QuantumCircuit = QuantumCircuit(self.num_qubits, len(self._cbit))
        qc.append(Initialize(pre_measure_state), range(self.num_qubits))
        qc.measure(qubit=self._qubit, cbit=self._cbit)
        result = backend.run(qc, shots=shots, seed_simulator=seed_simulator).result()
        postprocessed_result: R = self._postprocess(result)

        for name, condition in self._conditions_gen(param):
            if not (
                condition(
                    pre_measure_state,
                    result,
                    postprocessed_result,
                )
            ):
                raise err.MeasureConditionError(
                    err.MeasureConditionErrorInfo(
                        name, pre_measure_state, result, postprocessed_result, param
                    )
                )

        return postprocessed_result

    def run_without_assertion(
        self,
        shots: int,
        *,
        backend: AerBackend = AerSimulator(),
        init_state: (
            Statevector | OperatorBase | list[complex] | np.ndarray | None
        ) = None,
        seed_simulator: None | int = None,
    ) -> R:
        """Executes the quantum circuit, measures qubits,
        and postprocesses the measurement result without checking assertions.

        Args:
            shots (int):
                The number of repetitions of the circuit to be run.
            backend (AerBackend, optional):
                The backend to execute the circuit on. Defaults to AerSimulator().
            init_state
                (Statevector  |  OperatorBase  |  list[complex]  |  np.ndarray  |  None, optional):
                The initial state to run the circuit.
                If omitted, it is set to 0 ... 0.
            seed_simulator (None | int, optional):
                The seed of the simulator to fix the measurement result on each run.

        Returns:
            R: The postprocessed measurement result.
        """
        pre_measure_state: Statevector = self._aqc.run_without_assertion(
            init_state=init_state
        )
        qc: QuantumCircuit = QuantumCircuit(self.num_qubits, len(self._cbit))
        qc.append(Initialize(pre_measure_state), range(self.num_qubits))
        qc.measure(qubit=self._qubit, cbit=self._cbit)
        result = backend.run(qc, shots=shots, seed_simulator=seed_simulator).result()
        postprocessed_result: R = self._postprocess(result)
        return postprocessed_result

    def remove_assertions(self) -> tuple[QuantumCircuit, Callable[[Result], R]]:
        """Removes all assertions and returns a pair of
        the quantum circuit and the postprocessing.
        The quantum circuit is written in Qiskit,
        so you can run it on an actual quantum computer by Qiskit
        without providing the runtime parameters for the assertions.
        This method removes assertions from a copy of the instance of this class,
        so it does not affect the original instance.

        Returns:
            tuple[QuantumCircuit, Callable[[Result], R]]:
                The assertion-less quantum circuit and postprocessing.
        """
        qc: QuantumCircuit = QuantumCircuit(self.num_qubits, len(self._cbit))
        self._aqc._remove_assertions(qc, list(range(self.num_qubits)))
        qc.measure(qubit=self._qubit, cbit=self._cbit)
        return qc.decompose(utils._decompose_label), self._postprocess

    def remove_assertions_to_circuit(self) -> QuantumCircuit:
        """Removes all assertions and returns a QuantumCircuit that is written in Qiskit.
        You can run it on an actual quantum computer by Qiskit
        without providing the runtime parameters for the assertions.
        This method removes assertions from a copy of the circuit,
        so it does not affect the original circuit.

        Returns:
            QuantumCircuit: The assertion-less quantum circuit.
        """
        qc, _ = self.remove_assertions()
        return qc
