import itertools
import math

from qiskit.circuit.library.standard_gates import CXGate, HGate
from qiskit.opflow import One, Zero
from qiskit.quantum_info import Statevector

from dbcquantum.circuit import (
    AssertQuantumCircuit,
    CircuitCondition,
    PrePostCircuitCondition,
)
from dbcquantum.utils import eq_state, make_zeros_state, to_Statevector

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_infinite_condition1():
    aqc: AssertQuantumCircuit[None] = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append(CXGate(), [0, 1])

    def conditions(param: None) -> list[tuple[str, CircuitCondition]]:
        def condition(pre_state: Statevector, post_state: Statevector) -> bool:
            return False

        return list(itertools.repeat(("always false 1", condition)))

    def pre_post_conditions(param: None) -> list[tuple[str, PrePostCircuitCondition]]:
        def condition(state: Statevector) -> bool:
            return False

        return list(itertools.repeat(("always false 2", condition)))

    aqc.add_conditions_gen(conditions)
    aqc.add_conditions_gen(conditions)
    aqc.add_pre_conditions_gen(pre_post_conditions)
    aqc.add_pre_conditions_gen(pre_post_conditions)
    aqc.add_post_conditions_gen(pre_post_conditions)
    aqc.add_post_conditions_gen(pre_post_conditions)

    qc1 = aqc.remove_assertions()
    qc2 = aqc.inverse().inverse().inverse().inverse().remove_assertions()

    assert eq_state(make_zeros_state(2).evolve(qc1), bell)
    assert eq_state(make_zeros_state(2).evolve(qc2), bell)


def test_infinite_superpostion_1():
    aqc1: AssertQuantumCircuit[None] = AssertQuantumCircuit(2)
    aqc1.append(HGate(), [0])
    aqc1.append(CXGate(), [0, 1])

    def conditions(param: None) -> list[tuple[str, CircuitCondition]]:
        def condition(pre_state: Statevector, post_state: Statevector) -> bool:
            return False

        return list(itertools.repeat(("always false 1", condition)))

    def pre_post_conditions(param: None) -> list[tuple[str, PrePostCircuitCondition]]:
        def condition(state: Statevector) -> bool:
            return False

        return list(itertools.repeat(("always false 2", condition)))

    aqc1.add_conditions_gen(conditions)
    aqc1.add_conditions_gen(conditions)
    aqc1.add_pre_conditions_gen(pre_post_conditions)
    aqc1.add_pre_conditions_gen(pre_post_conditions)
    aqc1.add_post_conditions_gen(pre_post_conditions)
    aqc1.add_post_conditions_gen(pre_post_conditions)

    def pre_states_gen(
        pre_state: Statevector,
        param: None,
    ) -> list[tuple[complex, Statevector]]:
        return [(1, to_Statevector(Zero))] + list(
            itertools.repeat((0j, to_Statevector(Zero)))
        )

    aqc: AssertQuantumCircuit[None] = AssertQuantumCircuit(3)
    aqc.append_superposition_gen(
        "bell_super_position",
        aqc1,
        pre_superposition_states_gen=pre_states_gen,
        qargs=[0, 1],
    )

    qc1 = aqc.remove_assertions()
    qc2 = aqc.inverse().inverse().inverse().inverse().remove_assertions()

    assert eq_state(make_zeros_state(2).evolve(qc1), bell)
    assert eq_state(make_zeros_state(2).evolve(qc2), bell)
