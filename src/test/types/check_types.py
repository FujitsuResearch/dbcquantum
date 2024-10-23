import math
import typing

from qiskit.opflow import One, Zero
from qiskit.result.result import Result
from qiskit_aer import AerSimulator

from dbcquantum.circuit import AQCMeasure, AssertQuantumCircuit

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def type_check_1():
    def f(l: list[int | str]) -> int:
        return len(l)

    # f([0] + [1])

    S = typing.TypeVar("S", bound=(int | str))

    def g(l: list[S]) -> int:
        return len(l)

    l1: list[int] = [0] + [1]
    g(l1)

    l2: list[str] = ["a"] + ["b"]
    g(l2)

    l3: list[int | str] = [0] + ["b"]
    g(l3)

    # l4: list[complex] = [0j] + [1j]
    # g(l4)


def type_check_superposition_1():
    aqc1 = AssertQuantumCircuit(3)

    aqc2 = AssertQuantumCircuit(4)
    aqc2.append_superposition(
        "superposition1",
        aqc1,
        # pre_superposition_states: list[tuple[complex, DictStateFn | list[complex]]] | None
        pre_superposition_states=[
            (1 / math.sqrt(2) + 0j, Zero),
            (1 / math.sqrt(2) + 0j, [1j]),
        ],
        # post_superposition_states: list[tuple[complex, list[complex]]] | None
        post_superposition_states=[
            (1 / math.sqrt(2) + 0j, [0j]),
            (1 / math.sqrt(2) + 0j, [1j]),
        ],
        qargs=[3, 1, 0],
    )


def type_check_poly_aqc_1():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc2: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)
    aqc2.append(aqc1, [0])


def type_check_poly_aqc_2():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc2: AssertQuantumCircuit[str] = AssertQuantumCircuit(2)

    aqc2.append(aqc1, [0], param_converter=lambda h: len(h))


def type_check_poly_aqc_3():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc2: AssertQuantumCircuit[str] = AssertQuantumCircuit(2)
    aqc2.append(aqc1, [0], param_converter=lambda h: len(h))

    aqc2.run(param="aa")

    aqc3: AssertQuantumCircuit[None] = AssertQuantumCircuit(1)

    aqc3.append(aqc1, [2], param_converter=lambda h: 1)

    aqc3.run()


def type_check_inner_aqc():
    aqc1: AssertQuantumCircuit[int | str] = AssertQuantumCircuit(1)
    aqc2: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)
    aqc2.append(aqc1, [0])


def type_check_poly_shot_1():
    aqc: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)

    # aqc_shot0: AssertQuantumCircuitMeasure[None, Result] = aqc.prepare_measure(
    #     lambda result: result
    # )

    aqc_shot1: AQCMeasure[Result, None] = AQCMeasure(
        aqc, lambda result: result, param_converter=lambda _: 1
    )

    aqc_shot: AQCMeasure[Result, int] = AQCMeasure(aqc, lambda result: result)

    aqc_shot.add_condition(
        "condition",
        lambda pre_shot_state, result, counts: True,
    )

    shots = 10000
    simulator = AerSimulator()

    # aqc_shot.run(simulator, shots)
    aqc_shot.run(shots, param=1)
    aqc_shot1.run(shots, backend=simulator)


def check_types_superposition_1():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc2: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)

    aqc2.append_superposition_gen(
        "superposition1",
        aqc1,
        [0],
        pre_superposition_states_gen=lambda p, h: [
            (1 + 0j, Zero),
            (0 + 0j, One),
        ],
        post_superposition_states_gen=lambda p, h: [
            (1 + 0j, One),
            (0 + 0j, Zero),
        ],
    )


def check_types_superposition_2():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc2: AssertQuantumCircuit[str] = AssertQuantumCircuit(1)

    aqc2.append_superposition_gen(
        "superposition1",
        aqc1,
        pre_superposition_states_gen=lambda p, h: [
            (1 + 0j, Zero),
            (0 + 0j, One),
        ],
        post_superposition_states_gen=lambda p, h: [
            (1 + 0j, One),
            (0 + 0j, Zero),
        ],
        qargs=[0],
        param_converter=lambda h: [len(h), len(h)],
    )


def check_types_superposition_3():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)
    aqc2: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)

    aqc2.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 + 0j, Zero),
            (0 + 0j, One),
        ],
        post_superposition_states=[
            (1 + 0j, One),
            (0 + 0j, Zero),
        ],
        qargs=[0],
    )


def check_types_superposition_4():
    aqc1: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)

    aqc2: AssertQuantumCircuit[int] = AssertQuantumCircuit(1)

    aqc2.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 + 0j, Zero),
            (0 + 0j, One),
        ],
        post_superposition_states=[
            (1 + 0j, One),
            (0 + 0j, Zero),
        ],
        qargs=[0],
    )


def type_check_measure():
    aqc: AssertQuantumCircuit[int] = AssertQuantumCircuit(2)

    aqc_shot: AQCMeasure[int | str, None] = AQCMeasure(
        aqc, lambda result: "1", param_converter=lambda _: 1
    )

    aqc_shot.run(100)
