import math

import pytest
from qiskit import QiskitError, QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate, XGate
from qiskit.result.counts import Counts
from qiskit_aer import AerSimulator

from dbcquantum.circuit import AQCMeasure, AssertQuantumCircuit
from dbcquantum.utils import decompose


def test_decompose_1():

    aqc: AssertQuantumCircuit = AssertQuantumCircuit(2)
    aqc.append(XGate(), [0])
    aqc.append(
        decompose(RXGate(math.pi / 4 * 3).control(), ["rx", "rz", "h", "cx"]), [0, 1]
    )

    aqc_measure: AQCMeasure[Counts, None] = AQCMeasure(
        aqc, postprocess=lambda x: x.get_counts(), qubit=[1]  # type: ignore
    )

    aqc_measure.add_condition(
        "condition",
        lambda pre_measure_state, result, counts: counts.most_frequent() == "1",
    )

    aqc_measure.remove_assertions_to_circuit()
    aqc_measure.run(shots=1000)

    simulator = AerSimulator()
    qc, postprocess = aqc_measure.remove_assertions()
    result = simulator.run(qc, shots=1000).result()
    count = postprocess(result)
    assert count.most_frequent() == "1"


def test_decompose_2():

    aqc: AssertQuantumCircuit = AssertQuantumCircuit(2)

    qc = QuantumCircuit(1)
    qc.x(0)
    aqc.append(qc.to_instruction(), [0])
    # aqc.append(XGate(), [0])

    aqc.append(
        decompose(RXGate(math.pi / 4 * 3).control(), ["rx", "rz", "h", "cx"]), [0, 1]
    )

    aqc_measure: AQCMeasure[Counts, None] = AQCMeasure(
        aqc, postprocess=lambda x: x.get_counts(), qubit=[1]  # type: ignore
    )

    aqc_measure.add_condition(
        "condition",
        lambda pre_measure_state, result, counts: counts.most_frequent() == "1",
    )

    aqc_measure.remove_assertions_to_circuit()
    aqc_measure.run(shots=1000)

    simulator = AerSimulator()
    qc, postprocess = aqc_measure.remove_assertions()

    with pytest.raises(QiskitError):
        result = simulator.run(qc, shots=1000).result()
        postprocess(result)
