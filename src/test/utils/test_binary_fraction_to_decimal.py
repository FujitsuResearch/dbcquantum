import math

import dbcquantum.utils as utils


def test_binary_fraction_to_decimal_1():
    l = [1, 0, 1, 1]
    assert math.isclose(utils.bin_frac_to_dec(l, True), 0.6875)
    assert math.isclose(0.6875, 1 / 2 + 1 / 8 + 1 / 16)


def test_binary_fraction_to_decimal_2():
    l = [0, 0, 1, 1, 0, 1, 1, 0, 1]
    assert math.isclose(
        utils.bin_frac_to_dec(l, True),
        0 * (1 / 2**1)
        + 0 * (1 / 2**2)
        + 1 * (1 / 2**3)
        + 1 * (1 / 2**4)
        + 0 * (1 / 2**5)
        + 1 * (1 / 2**6)
        + 1 * (1 / 2**7)
        + 0 * (1 / 2**8)
        + 1 * (1 / 2**9),
    )
