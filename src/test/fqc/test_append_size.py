import pytest
from qiskit.circuit.library.standard_gates import CXGate, HGate

from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.err import DbCQuantumError


def test_append_size_1():
    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append(HGate(), [1])

    with pytest.raises(DbCQuantumError) as e:
        aqc.append(HGate(), [2])
    assert "The specified qubits are out of range!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc.append(HGate(), [-1])
    assert "The specified qubits are out of range!" in str(e.value)

    aqc2 = AssertQuantumCircuit(1)
    with pytest.raises(DbCQuantumError) as e:
        aqc2.append(aqc, [0, 1])
    assert "The instruction is bigger than the circuit!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc2.append(CXGate(), [0, 1])
    assert "The instruction is bigger than the circuit!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc.append(CXGate(), [0])
    assert "The length of qargs is inconsistent with the instruction!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc.append(CXGate(), [0, 1, 2])
    assert "The length of qargs is inconsistent with the instruction!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc.append(CXGate(), [1, 1])
    assert "duplicate qubit arguments" in str(e.value)


def test_append_size_2():
    aqc = AssertQuantumCircuit(2)
    aqc.append(HGate(), [0])
    aqc.append(HGate(), [1])

    aqc_h = AssertQuantumCircuit(1)
    aqc_h.append(HGate(), [0])

    aqc_cx = AssertQuantumCircuit(2)
    aqc_cx.append(CXGate(), [0, 1])

    with pytest.raises(DbCQuantumError) as e:
        aqc.append_superposition("aa", aqc_h, [2])
    assert "The specified qubits are out of range!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc.append_superposition("aa", aqc_h, [-1])
    assert "The specified qubits are out of range!" in str(e.value)

    aqc2 = AssertQuantumCircuit(1)
    with pytest.raises(DbCQuantumError) as e:
        aqc2.append_superposition("aa", aqc, [0, 1])
    assert "The instruction is bigger than the circuit!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc2.append_superposition("aa", aqc_cx, [0, 1])
    assert "The instruction is bigger than the circuit!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc.append_superposition("aa", aqc_cx, [0])
    assert "The length of qargs is inconsistent with the instruction!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc.append_superposition("aa", aqc_cx, [0, 1, 2])
    assert "The length of qargs is inconsistent with the instruction!" in str(e.value)

    with pytest.raises(DbCQuantumError) as e:
        aqc.append_superposition("aa", aqc_cx, [1, 1])
    assert "duplicate qubit arguments" in str(e.value)
