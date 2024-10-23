import cmath
import math
import typing

import numpy.linalg as LA
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library.standard_gates import HGate
from qiskit.opflow import H, One, OperatorBase, StateFn, T, Zero
from qiskit.quantum_info import Statevector

import dbcquantum.utils as utils
from dbcquantum.circuit import AssertQuantumCircuit
from dbcquantum.utils import eq_state

from .qft import make_qft


def make_qpe(
    U_gate: Gate, U: OperatorBase, n_precision: int
) -> AssertQuantumCircuit[tuple[float, list[int]]]:
    n_total: int = U_gate.num_qubits + n_precision
    circuit: AssertQuantumCircuit[tuple[float, list[int]]] = AssertQuantumCircuit(
        n_total
    )

    for i in range(n_precision):
        inner_circuit: AssertQuantumCircuit[list[int]] = AssertQuantumCircuit(
            1 + U_gate.num_qubits
        )
        inner_circuit.append(HGate(), [0])
        inner_circuit.append(
            U_gate.power(2**i).control(num_ctrl_qubits=1),
            range(1 + U_gate.num_qubits),
        )

        def inner_condition_fst_qubit(
            k: int,
        ) -> typing.Callable[[Statevector, list[int]], bool]:
            def condition(post_state: Statevector, param: list[int]) -> bool:
                js: list[int] = param
                a = cmath.exp(
                    1j * 2 * math.pi * utils.bin_frac_to_dec(js[k:n_precision], True)
                )
                state = 1 / math.sqrt(2) * (Zero + a * One)  # type: ignore
                return eq_state(utils.partial_state(post_state, [0]), state)

            return condition

        inner_circuit.add_post_condition_use_param(
            "inner_condition_fst_qubit: qubit" + str(i), inner_condition_fst_qubit(i)
        )

        inner_circuit.add_condition(
            "inner_condition do not change state" + str(i),
            lambda pre_state, post_state: eq_state(
                utils.partial_state(pre_state, range(1, 1 + U_gate.num_qubits)),
                utils.partial_state(post_state, range(1, 1 + U_gate.num_qubits)),
            ),
        )

        circuit.append(
            inner_circuit,
            [i] + list(range(n_precision, n_precision + U_gate.num_qubits)),
            param_converter=lambda h: h[1],
        )

    circuit.append(
        make_qft(n_precision).inverse(),
        range(n_precision),
    )

    circuit.add_pre_condition(
        "[qpe] all input qubits for answer must be |0>",
        lambda pre_state: all(
            eq_state(s, Zero) or eq_state(s, One)
            for s in utils.split_each_qubit_states(
                utils.partial_state(pre_state, range(n_precision))
            )
        ),
    )

    def param_is_eigenvector(pre_state: Statevector, param: tuple[float, list[int]]):
        atol, js = param
        psi: StateFn = StateFn(
            utils.partial_state(
                pre_state, range(n_precision, n_precision + U_gate.num_qubits)
            )
        )

        eigen = cmath.exp(1j * 2 * math.pi * utils.bin_frac_to_dec(js, True))
        return eq_state(U @ psi, eigen * psi, atol=atol)  # type: ignore

    circuit.add_pre_condition_use_param(
        "[qpe] param is eigenvector",
        param_is_eigenvector,
    )

    circuit.add_condition(
        "[qpe] don't change psi",
        lambda pre_state, post_state: eq_state(
            utils.partial_state(
                pre_state, range(n_precision, n_precision + U_gate.num_qubits)
            ),
            utils.partial_state(
                post_state, range(n_precision, n_precision + U_gate.num_qubits)
            ),
        ),
    )

    def post_state_is_eigen_vector(
        pre_state, post_state: Statevector, param: tuple[float, list[int]]
    ) -> bool:
        atol, _ = param
        psi: StateFn = StateFn(
            utils.partial_state(
                pre_state, range(n_precision, n_precision + U_gate.num_qubits)
            )
        )

        state_js = [
            0 if eq_state(state, Zero) else 1
            for state in utils.split_each_qubit_states(
                utils.partial_state(post_state, range(n_precision))
            )
        ]

        eigen = cmath.exp(1j * 2 * math.pi * utils.bin_frac_to_dec(state_js, True))
        return eq_state(U @ psi, eigen * psi, atol=atol)  # type: ignore

    circuit.add_condition_use_param(
        "[qpe] post state is eigen vector", post_state_is_eigen_vector
    )

    return circuit


def fraction_to_binary(a: float, n: int) -> list[int]:
    assert a < 1
    ans = []

    for i in range(1, n + 1):
        if (1 / (2**i)) <= a:
            ans.append(1)
            a -= 1 / (2**i)
        else:
            ans.append(0)

    return ans


def test_qpe_1():
    n_precision: int = 5

    U: OperatorBase = T ^ H  # type: ignore
    c = QuantumCircuit(2)
    c.t(1)
    c.h(0)
    U_gate: Gate = c.to_gate()

    eigen = LA.eig(U.to_matrix())

    for i in range(len(eigen[0])):
        # calculate eigenvalue and eigenstate
        eigenstate = StateFn(eigen[1][:, i])
        eigenvalue = eigen[0][i]

        print("actual_eigen =", eigenvalue)

        # e^(i*2pi*phi) = eigenvalue
        # phi = log(eigenvalue) / i * 2pi
        phi: float = (cmath.log(eigenvalue) / (1j * 2 * math.pi)).real

        if phi < 0:
            phi = phi + 1

        assert phi >= 0

        # check (cmath.log(eigenvalue) / (1j * 2 * math.pi)) is a real number
        assert cmath.isclose(
            cmath.exp(1j * 2 * math.pi * phi), eigenvalue, abs_tol=1e-08
        )

        print("phi =", phi)
        js: list[int] = fraction_to_binary(phi, n_precision)
        print("js =", js)

        init_state: OperatorBase = Zero
        for i in range(n_precision - 1):
            init_state = typing.cast(OperatorBase, init_state ^ Zero)
        init_state = eigenstate ^ init_state  # type: ignore

        circuit = make_qpe(U_gate, U, n_precision)
        state = circuit.run(init_state=init_state, param=(1e-1, js))

        state_js: list[int] = [
            0 if eq_state(state, Zero) else 1
            for state in utils.split_each_qubit_states(
                utils.partial_state(state, range(n_precision))
            )
        ]

        assert state_js == js
