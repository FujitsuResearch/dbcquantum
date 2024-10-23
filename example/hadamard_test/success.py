import cmath
import math

from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import HGate, TGate
from qiskit.opflow import One, OperatorBase, Plus, StateFn, Zero
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.operator import Operator

try:
    from dbcquantum.circuit import AssertQuantumCircuit
    from dbcquantum.utils import decompose, eq_state, partial_state
except ImportError:
    # Development
    import sys

    sys.path.append("src")
    from dbcquantum.circuit import AssertQuantumCircuit
    from dbcquantum.utils import decompose, eq_state, partial_state


def make_circ_hadamal_test(U_gate: Gate, U: OperatorBase) -> AssertQuantumCircuit:

    circ = AssertQuantumCircuit(U_gate.num_qubits + 1)
    circ.append(HGate(), [0])
    ctrl_U = decompose(U_gate.control(), basis_gates=["h", "rx", "rz", "cx"])
    circ.append(ctrl_U, range(U_gate.num_qubits + 1))
    circ.append(HGate(), [0])

    def condition(pre_state: Statevector, post_state: Statevector) -> bool:

        psi = StateFn(partial_state(pre_state, range(1, U_gate.num_qubits + 1)))

        state_0 = ((psi + (U @ psi)) / 2) ^ Zero  # type: ignore
        state_1 = ((psi - (U @ psi)) / 2) ^ One  # type: ignore
        return eq_state(post_state, state_0 + state_1)

    circ.add_condition("condition1", condition)
    return circ


# U: OperatorBase = T
U: OperatorBase = PrimitiveOp(Operator([[1, 0], [0, cmath.exp(1j * math.pi / 4)]]))
U_gate: Gate = TGate()
circ_hadamard_test = make_circ_hadamal_test(U_gate, U)

psi = Plus
print(circ_hadamard_test.run(init_state=psi ^ Zero))
