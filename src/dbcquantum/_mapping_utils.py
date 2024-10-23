from .common import QubitMapping, QubitSpecifier


def apply_all(
    global_mapping: QubitMapping, qargs: list[QubitSpecifier]
) -> list[QubitSpecifier]:
    return list(map(lambda x: global_mapping[x], qargs))


def compose(mapping1: QubitMapping, mapping2: QubitMapping) -> QubitMapping:
    mapping = [0] * len(mapping2)

    for i in range(len(mapping2)):
        mapping[i] = mapping1[mapping2[i]]

    return mapping
