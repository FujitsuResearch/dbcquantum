import math

import pandas as pd
import pytest
from qiskit.circuit.library.standard_gates import CXGate, HGate, RXGate, XGate
from qiskit.opflow import One, Plus, Zero
from qiskit.quantum_info import Statevector
from qiskit.result.counts import Counts
from qiskit.result.result import Result
from qiskit_aer import AerSimulator
from scipy.stats import chisquare

from dbcquantum.circuit import AQCMeasure, AssertQuantumCircuit
from dbcquantum.err import MeasureConditionError
from dbcquantum.utils import eq_state

seed_simulator = 10


def test_measure_1():
    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append(HGate(), [1])

    assert eq_state(aqc.run(), Plus ^ Plus)  # type: ignore

    shots = 10000
    # simulator = Aer.get_backend("aer_simulator")
    aqc_measure: AQCMeasure[Counts, None] = AQCMeasure(
        aqc, lambda result: result.get_counts()  # type: ignore
    )

    def condition(
        pre_measure_state: Statevector, result: Result, counts: Counts
    ) -> bool:
        observed_counts = dict(counts.items())

        expected_counts = {
            "00": shots / 4,
            "01": shots / 4,
            "10": shots / 4,
            "11": shots / 4,
        }

        d = {"observed_counts": observed_counts, "expected_counts": expected_counts}
        df = pd.DataFrame(d).fillna(0).sort_index()

        p = chisquare(
            df["observed_counts"].to_list(), df["expected_counts"].to_list()
        ).pvalue

        return not (p < 0.05)

    aqc_measure.add_condition("condition", condition)
    aqc_measure.run(shots=shots, param=None, seed_simulator=seed_simulator)


def test_measure_2():
    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append(HGate(), [1])

    assert eq_state(aqc.run(), Plus ^ Plus)  # type: ignore

    shots = 10000
    simulator = AerSimulator()
    aqc_measure: AQCMeasure[Counts, None] = AQCMeasure(
        aqc, lambda result: result.get_counts()  # type: ignore
    )

    def condition(
        pre_measure_state: Statevector, result: Result, counts: Counts
    ) -> bool:
        observed_counts = dict(counts.items())

        expected_counts = {
            "00": shots / 2,
            "01": shots / 6,
            "10": shots / 6,
            "11": shots / 6,
        }

        d = {"observed_counts": observed_counts, "expected_counts": expected_counts}
        df = pd.DataFrame(d).fillna(0).sort_index()

        p = chisquare(
            df["observed_counts"].to_list(), df["expected_counts"].to_list()
        ).pvalue

        return not (p < 0.05)

    aqc_measure.add_condition("condition", condition)

    with pytest.raises(MeasureConditionError) as e:
        aqc_measure.run(shots, backend=simulator, seed_simulator=seed_simulator)

    assert eq_state(e.value.info.pre_measure_state, Plus ^ Plus)  # type: ignore
    assert "Condition Error occured in 'condition'" in str(e.value)

    assert (
        "condition_name='condition'\n"
        "pre_measure_state=\n"
        "  Statevector([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j],\n"
        "              dims=(2, 2))\n" in str(e.value)
    )
    assert "result=Result" in str(e.value)
    assert "param=None" in str(e.value)


def test_measure_3():
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

    aqc_measure.run(shots, seed_simulator=seed_simulator)


bell = 1 / math.sqrt(2) * ((Zero ^ Zero) + (One ^ One))  # type: ignore


def test_focus_qubit_of_add_condition():
    aqc = AssertQuantumCircuit(3)

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    aqc.append(make_bell, [1, 2])
    aqc.append(XGate(), [0])

    aqc_measure: AQCMeasure[Counts, None] = AQCMeasure(
        aqc, lambda result: result.get_counts(), qubit=[1, 0]  # type: ignore
    )

    aqc_measure.add_condition(
        "condition1",
        lambda pre_measure_state, result, counts: eq_state(pre_measure_state, bell),
        focus_qubits=[1, 2],
    )

    aqc_measure.run(1000)


def test_focus_qubit_of_add_condition_fail():
    aqc = AssertQuantumCircuit(3)

    make_bell = AssertQuantumCircuit(2)
    make_bell.append(HGate(), [0])
    make_bell.append(CXGate(), [0, 1])

    aqc.append(make_bell, [1, 2])
    aqc.append(XGate(), [0])

    aqc_measure: AQCMeasure[Counts, None] = AQCMeasure(
        aqc, lambda result: result.get_counts(), qubit=[1, 0]  # type: ignore
    )

    aqc_measure.add_condition(
        "condition1",
        lambda pre_measure_state, result, counts: eq_state(
            pre_measure_state, Zero ^ One  # type: ignore
        ),
        focus_qubits=[1, 2],
    )

    with pytest.raises(MeasureConditionError) as e:
        aqc_measure.run(1000)

    assert "Condition Error occured in 'condition1'" in str(e.value)
