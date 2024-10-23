from typing import Callable, Iterable, TypeVar, cast

import numpy as np
import qiskit.compiler.transpiler as transpiler
from qiskit import QiskitError, QuantumCircuit, execute
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library import Permutation
from qiskit.opflow.operator_base import OperatorBase
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit_aer import AerSimulator

from . import err
from .common import QubitSpecifier

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def eq_state(
    s1: Statevector | OperatorBase | list[complex] | np.ndarray,
    s2: Statevector | OperatorBase | list[complex] | np.ndarray,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> bool:
    """Return True if other is equivalent as a statevector up to global phase.

    Args:
        s1 (Statevector | OperatorBase | list[complex] | np.ndarray):
            The first state to be compared.
        s2 (Statevector | OperatorBase | list[complex] | np.ndarray):
            The second state to be compared.
        rtol (float | None, optional): rtol (float):
            relative tolerance value for comparison.
        atol (float | None, optional): atol (float):
            absolute tolerance value for comparison.

    Returns:
        bool: True if statevectors are equivalent up to global phase.
    """
    _s1: Statevector = to_Statevector(s1)
    _s2: Statevector = to_Statevector(s2)

    if _s1.num_qubits != _s2.num_qubits:
        raise err.DbCQuantumError(
            "The states cannot be compared with because num_qubits is not the same!"
        )

    return _s1.equiv(_s2, rtol=rtol, atol=atol)


def to_Statevector(
    s: Statevector | OperatorBase | list[complex] | np.ndarray,
) -> Statevector:
    """Convert to Statevector

    Args:
        s (Statevector | OperatorBase | list[complex] | np.ndarray):
            Statevector compatible input

    Returns:
        Statevector: Converted Statevector
    """
    match s:
        case Statevector():
            return s
        case OperatorBase():
            return Statevector(s.to_matrix(massive=True))
        case _:
            return Statevector(s)


def _check_index(i: int, qubits: list[int], outcome: list[int]) -> bool:
    for q, r in zip(qubits, outcome):
        if _get_nth_bit(i, q) != r:
            return False
    return True


def _get_nth_bit(value, n) -> int:
    return (value >> n) & 1


# def partial_state_traceout(
#     state: Statevector,
#     trace_out: list[QubitSpecifier],
#     atol: float | None = None,
#     rtol: float | None = None,
# ) -> Statevector:
#     state = to_Statevector(state)
#     state = state.copy()

#     if len(trace_out) == 0:
#         return state

#     trace_out = sorted(set(trace_out))
#     r: tuple[list[str], Statevector] = state.measure(trace_out)  # type: ignore
#     (_outcome, state_after_measure) = r
#     vec = state_after_measure.data
#     outcome: list[int] = list(reversed(list(map(int, _outcome))))
#     index = [i for i in range(vec.size) if _check_index(i, trace_out, outcome)]
#     ret = Statevector(vec[index])

#     # if not ret.is_valid(atol, rtol):
#     #     raise err.DbCQuantumError(
#     #         "The specified qubits are entangled with the other qubits!"
#     #     )

#     return ret

backend = AerSimulator(method="statevector")


def _partial_state_traceout(
    state: Statevector,
    trace_out: list[QubitSpecifier],
    atol: float | None = None,
    rtol: float | None = None,
) -> Statevector:
    """Extract the state of some qubits from the entire quantum state.
    You can specify the qubits that are not needed.
    The entanglement is not checked.

    Args:
        state (Statevector): The entire quantum state.
        trace_out (list[QubitSpecifier]): List of qubits to be traced out.

    Returns:
        Statevector: The state of the remaining qubits after trace out.


    Examples:
        >>> print(eq_state(Plus ^ Zero,
                           _partial_state_traceout(
                                to_Statevector(Minus ^ Plus ^ One ^ Zero),
                                [3, 1])))
        True
    """

    if len(trace_out) == 0:
        return state

    if any(map(lambda q: q < 0 or q >= state.num_qubits, trace_out)):
        raise err.DbCQuantumError("The specified qubits are out of range!")

    trace_out = sorted(set(trace_out))

    qc = QuantumCircuit(state.num_qubits, len(trace_out))
    qc.set_statevector(state.data)  # type: ignore
    qc.measure(trace_out, range(len(trace_out)))
    qc.save_statevector(label="test", pershot=True)  # type: ignore

    result = execute(qc, backend=backend, shots=1).result()
    _outcome: str = list(result.get_counts().items())[0][0]
    state_after_measure = result.data(0)["test"][0]

    vec = state_after_measure.data
    outcome: list[int] = list(reversed(list(map(int, _outcome))))
    index = [i for i in range(vec.size) if _check_index(i, trace_out, outcome)]
    ret = Statevector(vec[index])

    if not ret.is_valid(atol, rtol):
        raise err.DbCQuantumError(
            "The specified qubits are entangled with the other qubits!"
        )
    return ret


def _get_order(l: list[int]) -> list[int]:
    sorted_qubits = sorted(l)
    return [sorted_qubits.index(i) for i in l]


def partial_state(
    state: Statevector, focus_qubits: Iterable[QubitSpecifier]
) -> Statevector:
    """Extract the state of some qubits from the entire quantum state.
    You can specify the qubits whose state you want to know.

    Args:
        state (Statevector): The entire quantum state.
        focus_qubits (Iterable[QubitSpecifier]):
            Qubits whose state is to be extracted.
            The order is reflected in the returned statevector.

    Returns:
        Statevector: The state of the specified qubits.


    Examples:
        >>> print(eq_state(One ^ Minus,
                           partial_state(
                                to_Statevector(Minus ^ Plus ^ One ^ Zero),
                                [3, 1])))
        True
    """
    _focus_qubits: list[QubitSpecifier] = list(focus_qubits)

    if any(map(lambda q: q < 0 or q >= state.num_qubits, _focus_qubits)):
        raise err.DbCQuantumError("The specified qubits are out of range!")

    trace_out: list[QubitSpecifier] = [
        i for i in range(state.num_qubits) if i not in _focus_qubits
    ]

    focus_state = _sort_statevector(
        _partial_state_traceout(state, trace_out), _get_order(_focus_qubits)
    )
    other_state = _partial_state_traceout(state, _focus_qubits)
    entire_state = _merge_statevector(focus_state, other_state, _focus_qubits)

    if not eq_state(state, entire_state):
        raise err.DbCQuantumError(
            "The specified qubits are entangled with the other qubits!"
        )

    return focus_state


def _sort_statevector(vec: Statevector, order: list[int]) -> Statevector:
    if vec.num_qubits != len(order):
        raise err.DbCQuantumError("The size of vec and mapping is inconsistent!")

    if vec.num_qubits == 0 or vec.num_qubits == 1:
        return vec

    v = vec.evolve(Permutation(vec.num_qubits, order), qargs=order)
    return v


def _inverse_order_for_sort_statevector(order: list[int]) -> list[int]:
    inverse_order: list[int] = [0] * len(order)
    for i in range(len(order)):
        inverse_order[order[i]] = i
    return inverse_order


def _merge_statevector(
    focus_qubits_state: Statevector,
    trace_out_state: Statevector,
    focus_qubits: list[QubitSpecifier],
):
    if focus_qubits_state.num_qubits != len(focus_qubits):
        raise err.DbCQuantumError(
            "The size of focus_qubits_state and focus_qubits is inconsistent!"
        )

    trace_out_qubits: list[QubitSpecifier] = [
        i
        for i in range(focus_qubits_state.num_qubits + trace_out_state.num_qubits)
        if i not in focus_qubits
    ]

    state: Statevector = trace_out_state.tensor(focus_qubits_state)
    return _sort_statevector(
        state, _inverse_order_for_sort_statevector(focus_qubits + trace_out_qubits)
    )


def split_each_qubit_states(
    state: Statevector, reverse: bool = False
) -> list[Statevector]:
    """Split the quantum state into individual states of each qubit.

    Args:
        state (Statevector): The entire quantum state.
        reverse (bool, optional): If True, reverse the order of the qubits. Defaults to False.

    Returns:
        list[Statevector]: List of statevectors representing the state of each qubit.
    """
    ret: list[Statevector] = []

    for i in range(state.num_qubits):
        i_bit_state: Statevector = partial_state(state, [i])
        ret.append(i_bit_state)

    if reverse:
        return list(reversed(ret))
    else:
        return ret


def make_zeros_state(num_qubits: int) -> Statevector:
    """Create a statevector representing a quantum state with all qubits in the zero state.

    Args:
        num_qubits (int): The number of qubits for the state.

    Returns:
        Statevector: A statevector representing a quantum state with all qubits in the zero state.
    """
    vec: list[float] = [0] * (2**num_qubits)
    vec[0] = 1
    return Statevector(vec)


def state_sum(
    states: Iterable[tuple[complex, Statevector]], num_qubits: int
) -> Statevector:
    """Calculate the sum of given quantum states.

    Args:
        states (Iterable[Tuple[complex, Statevector]]):
            An iterable of tuples, each containing a complex coefficient and a statevector.
        num_qubits (int):
            The number of qubits for the state.

    Returns:
        Statevector: The resulting statevector after summing the given states.

    Examples:
        >>> print(eq_state(Plus, state_sum([(1/sqrt(2), Zero), (1/sqrt(2), One)], 1)))
        True
    """
    sum_states = np.zeros(2**num_qubits, dtype=complex)

    for a, state in states:
        sum_states = sum_states + (a * state.data)

    return Statevector(sum_states)


def _lazy_composition(
    f: Callable[[S], list[T]], g: Callable[[S], list[T]]
) -> Callable[[S], list[T]]:
    def h(x: S) -> list[T]:
        return f(x) + g(x)

    return h


def _lazy_map(
    f: Callable[[S], T], lazy_l: Callable[[U], list[S]]
) -> Callable[[U], list[T]]:
    def mapped_lazy_l(u: U) -> list[T]:
        return list(map(f, lazy_l(u)))

    return mapped_lazy_l


def _lazy_map2(
    f: Callable[[S], T], lazy_l: Callable[[U, V], list[S]]
) -> Callable[[U, V], list[T]]:
    def mapped_lazy_l(u: U, v: V) -> list[T]:
        return list(map(f, lazy_l(u, v)))

    return mapped_lazy_l


def bin_frac_to_dec(
    binary_fraction: list[int] | list[bool], reverse: bool = False
) -> float:
    """Convert a binary fraction to a decimal number.

    Args:
        binary_fraction (Union[List[int], List[bool]]):
            The binary fraction to be converted.
        reverse (bool, optional):
            If True, reverse the order of the binary fraction before conversion.
            Defaults to False.

    Returns:
        float: The decimal representation of the binary fraction.

    Examples:
        >>> print(binary_fraction_to_decimal([1,0,1,1], reverse = False))
        0.8125

        >>> print(binary_fraction_to_decimal([1,0,1,1], reverse = True))
        0.6875
    """
    a: float = 0.0

    if reverse:
        _binary_fraction = reversed(binary_fraction)
    else:
        _binary_fraction = binary_fraction

    for i in _binary_fraction:
        a += i
        a /= 2
    return a


_decompose_label: str = "_decompose"


def decompose(instraction: Instruction, basis_gates: list[str]) -> Instruction:
    """Decompose a quantum instruction into a sequence of instructions
    from a specified set of basis gates.

    Args:
        instruction (Instruction): The quantum instruction to be decomposed.
        basis_gates (List[str]): The set of basis gates to be used for the decomposition.

    Returns:
        Instruction: The decomposed instruction as a sequence of basis gates.
    """
    qc = QuantumCircuit(instraction.num_qubits)
    qc.append(instraction, range(instraction.num_qubits))
    transpiled = cast(QuantumCircuit, transpiler.transpile(qc, basis_gates=basis_gates))
    return transpiled.to_instruction(label=_decompose_label)


def density_matrix_to_statevector(
    dt: DensityMatrix, atol=None, rtol=None
) -> Statevector:
    """Return a statevector from a pure density matrix.

    Args:
        atol (float): Absolute tolerance for checking operation validity.
        rtol (float): Relative tolerance for checking operation validity.

    Returns:
        Statevector: The pure density matrix's corresponding statevector.
            Corresponds to the eigenvector of the only non-zero eigenvalue.

    Raises:
        QiskitError: if the state is not pure.
    """
    from qiskit.quantum_info.states.statevector import Statevector

    if atol is None:
        atol = dt.atol
    if rtol is None:
        rtol = dt.rtol

    if not is_hermitian_matrix(dt._data, atol=atol, rtol=rtol):
        raise QiskitError("Not a valid density matrix (non-hermitian).")

    evals, evecs = np.linalg.eigh(dt._data)

    nonzero_evals = evals[abs(evals) > atol]
    if len(nonzero_evals) != 1 or not np.isclose(
        nonzero_evals[0], 1, atol=atol, rtol=rtol
    ):
        raise QiskitError("Density matrix is not a pure state")

    psi = evecs[:, np.argmax(evals)]  # eigenvectors returned in columns.
    return Statevector(psi)


def is_basis(basis: list[Statevector]) -> bool:
    """Check if the given set of statevectors forms a basis.

    Args:
        basis (List[Statevector]): The set of statevectors to be checked.

    Returns:
        bool: True if the set forms a basis, False otherwise.
    """
    if len(basis) == 0:
        raise RuntimeError("basis is empty.")

    if len(set(map(lambda s: s.num_qubits, basis))) != 1:
        raise RuntimeError("The size of elements in basis is inconsistent.")

    if len(basis) != 2 ** basis[0].num_qubits:
        raise RuntimeError("The size of basis is too small.")

    basis_vectors = [s.data for s in basis]

    matrix = np.vstack(basis_vectors)
    rank = np.linalg.matrix_rank(matrix)

    if rank == 2 ** basis[0].num_qubits:
        return True
    else:
        return False


def coeff_of_basis(vec: Statevector, basis: list[Statevector]) -> list[complex]:
    """Calculate the coefficients of the given vector in the given basis.

    Args:
        vec (Statevector): The vector whose coefficients are to be calculated.
        basis (List[Statevector]): The basis in which to express the vector.

    Returns:
        List[complex]: The coefficients of the vector in the given basis.
    """
    if vec.num_qubits != basis[0].num_qubits:
        raise RuntimeError("The size of vector and basis is inconsistent.")

    matrix = np.vstack(basis).T
    result = np.linalg.lstsq(matrix, vec, rcond=None)
    return list(result[0])


def binary_basis(num_qubits: int) -> list[Statevector]:
    """Generate a binary basis for the given number of qubits.

    Args:
        num_qubits (int): The number of qubits for which to generate the basis.

    Returns:
        List[Statevector]: The binary basis for the given number of qubits.
    """
    n = 2**num_qubits
    identity = np.eye(n)
    return [Statevector(identity[i]) for i in range(n)]


def is_scalar_multiple(u: np.ndarray, v: np.ndarray) -> bool:
    """Check whether u = kv for some scalar k.

    Args:
        u (np.ndarray): Vector.
        v (np.ndarray): Vector.

    Returns:
        bool: Result.
    """
    # avoiding division by zero.
    if not np.array_equiv(np.where(u == 0), np.where(v == 0)):
        return False

    index = np.where(v != 0)
    k_list = u[index] / v[index]

    k = np.unique(k_list)
    return k.size == 1
