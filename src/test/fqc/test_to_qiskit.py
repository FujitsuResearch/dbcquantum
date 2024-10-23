import math

from qiskit.circuit.library.standard_gates import CXGate, HGate, RXGate, XGate
from qiskit.opflow import One, Plus, X, Zero
from qiskit.result.counts import Counts
from qiskit_aer import AerSimulator

from dbcquantum.circuit import AQCMeasure, AssertQuantumCircuit
from dbcquantum.utils import eq_state, make_zeros_state, partial_state

bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_to_qiskit_1():
    aqc1 = AssertQuantumCircuit(1)
    aqc2 = AssertQuantumCircuit(2)

    aqc1.append(HGate(), [0])
    aqc2.append(CXGate(), [0, 1])

    aqc = AssertQuantumCircuit(2)
    aqc.append(aqc1, [0])
    aqc.append(aqc2, [0, 1])

    qc = aqc.remove_assertions()
    state = aqc.run()

    assert eq_state(state, bell)
    assert eq_state(state, make_zeros_state(2).evolve(qc))


def test_to_qiskit_2():
    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    make_bell.add_condition(
        "bell_input_|00>",
        lambda pre_state, post_state: eq_state(pre_state, Zero ^ Zero),  # type: ignore
    )

    make_bell.add_condition(
        "bell_output_bell",
        lambda pre_state, post_state: eq_state(post_state, bell),  # type: ignore
    )

    swap = AssertQuantumCircuit(2)
    swap.append(CXGate(), [0, 1])
    swap.append(CXGate(), [1, 0])
    swap.append(CXGate(), [0, 1])

    swap.add_condition(
        "swap_condition",
        lambda pre_state, post_state: eq_state(
            partial_state(pre_state, [0]), partial_state(post_state, [1])
        )
        and eq_state(partial_state(pre_state, [1]), partial_state(post_state, [0])),
    )

    # HZH = X
    x = AssertQuantumCircuit(1)
    x.append(XGate(), [0])
    x.add_condition(
        "x_condition",
        lambda pre_state, post_state: eq_state(post_state, pre_state.evolve(X)),
    )

    aqc1 = AssertQuantumCircuit(3)
    aqc1.append(XGate(), [0])
    aqc1.append(HGate(), [2])
    aqc1.append(make_bell, [0, 2])
    aqc1.append(XGate(), [1])

    aqc1.add_condition(
        "aqc1_condition1",
        lambda pre_state, post_state: eq_state(partial_state(pre_state, [0]), One),
    )
    aqc1.add_condition(
        "aqc1_condition2",
        lambda pre_state, post_state: eq_state(partial_state(pre_state, [1]), Zero),
    )
    aqc1.add_condition(
        "aqc1_condition3",
        lambda pre_state, post_state: eq_state(partial_state(pre_state, [2]), Plus),
    )
    aqc1.add_condition(
        "aqc1_condition4",
        lambda pre_state, post_state: eq_state(partial_state(post_state, [1]), One),
    )
    aqc1.add_condition(
        "aqc1_condition5",
        lambda pre_state, post_state: eq_state(partial_state(post_state, [0, 2]), bell),
    )

    aqc = AssertQuantumCircuit(4)
    aqc.append(x, [3])
    assert eq_state(aqc.run(), One ^ Zero ^ Zero ^ Zero)  # type: ignore

    aqc.append(HGate(), [0])
    assert eq_state(aqc.run(), One ^ Zero ^ Zero ^ Plus)  # type: ignore

    aqc.append(aqc1, [3, 1, 0])
    assert eq_state(partial_state(aqc.run(), [3, 0]), bell)
    assert eq_state(partial_state(aqc.run(), [1]), One)

    aqc.append(swap, [2, 3])
    aqc.append(swap, [1, 2])

    qc = aqc.remove_assertions()
    assert eq_state(make_zeros_state(4).evolve(qc), Zero ^ One ^ bell)


def test_superposition_example_to_qiskit():
    aqc1 = AssertQuantumCircuit(2)
    aqc1.append(CXGate(), [0, 1])
    aqc1.add_pre_condition(
        "pre_condition1",
        lambda pre_condition: eq_state(pre_condition, Zero ^ Zero)  # type: ignore
        or eq_state(pre_condition, Zero ^ One),  # type: ignore
    )
    aqc1.add_post_condition(
        "post_condition1",
        lambda post_condition: eq_state(post_condition, Zero ^ Zero)  # type: ignore
        or eq_state(post_condition, One ^ One),  # type: ignore
    )

    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])

    aqc.append_superposition(
        "superposition1",
        aqc1,
        pre_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), Zero ^ One),  # type: ignore
        ],
        post_superposition_states=[
            (1 / math.sqrt(2), Zero ^ Zero),
            (1 / math.sqrt(2), One ^ One),  # type: ignore
        ],
        qargs=[0, 1],
    )

    state = aqc.run()
    assert eq_state(state, bell)

    qc = aqc.inverse().inverse().remove_assertions()
    assert eq_state(make_zeros_state(2).evolve(qc), bell)


def test_to_qiskit_measure():
    aqc = AssertQuantumCircuit(2)
    aqc.append(RXGate(math.pi * 1 / 6), [0])
    aqc.append(RXGate(math.pi * 5 / 6), [1])

    aqc_measure: AQCMeasure[Counts, None] = AQCMeasure(
        aqc, lambda result: result.get_counts(), qubit=[1, 0]  # type: ignore
    )

    aqc_measure.add_condition(
        "condition",
        lambda pre_measure_state, result, counts: counts.most_frequent() == "01",
    )
    shots = 10000
    simulator = AerSimulator()

    aqc_measure.run(shots)

    qc, postprocess = aqc_measure.remove_assertions()
    result = simulator.run(qc, shots=shots).result()
    count = postprocess(result)
    assert count.most_frequent() == "01"
