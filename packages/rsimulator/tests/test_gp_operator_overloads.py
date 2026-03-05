import unittest

from rsimulator import (
    flat_gp_const,
    flat_gp_add,
    flat_gp_sub,
    flat_gp_mul,
    flat_gp_div,
    flat_gp_travel_time,
    flat_gp_time_until_due,
)


class TestFlatGpTreeOperatorOverloads(unittest.TestCase):
    def _ops(self, tree):
        return bytes(tree.to_bytes())

    def test_add_sub_mul_div_neg_equivalence(self):
        # add
        a = flat_gp_const(1.0) + flat_gp_const(2.0)
        expect_add = flat_gp_add(flat_gp_const(1.0), flat_gp_const(2.0))
        self.assertEqual(self._ops(a), self._ops(expect_add))

        # sub (left - right)
        s = flat_gp_const(0.0) - flat_gp_travel_time()
        expect_sub = flat_gp_sub(flat_gp_const(0.0), flat_gp_travel_time())
        self.assertEqual(self._ops(s), self._ops(expect_sub))

        # negation
        n = -flat_gp_travel_time()
        expect_neg = flat_gp_sub(flat_gp_const(0.0), flat_gp_travel_time())
        self.assertEqual(self._ops(n), self._ops(expect_neg))

        # mul
        m = flat_gp_const(2.0) * flat_gp_const(3.0)
        expect_mul = flat_gp_mul(flat_gp_const(2.0), flat_gp_const(3.0))
        self.assertEqual(self._ops(m), self._ops(expect_mul))

        # div (protected)
        d = flat_gp_const(1.0) / flat_gp_const(0.0)
        expect_div = flat_gp_div(flat_gp_const(1.0), flat_gp_const(0.0))
        self.assertEqual(self._ops(d), self._ops(expect_div))

    def test_numeric_coercion_and_r_ops(self):
        # numeric left (radd/rsub/rmul/rdiv)
        radd = 5.0 + flat_gp_const(3.0)
        expect_radd = flat_gp_add(flat_gp_const(5.0), flat_gp_const(3.0))
        self.assertEqual(self._ops(radd), self._ops(expect_radd))

        rsub = 5.0 - flat_gp_const(3.0)
        expect_rsub = flat_gp_sub(flat_gp_const(5.0), flat_gp_const(3.0))
        self.assertEqual(self._ops(rsub), self._ops(expect_rsub))

        rmul = 2.0 * flat_gp_const(4.0)
        expect_rmul = flat_gp_mul(flat_gp_const(2.0), flat_gp_const(4.0))
        self.assertEqual(self._ops(rmul), self._ops(expect_rmul))

        rdiv = 6.0 / flat_gp_const(2.0)
        expect_rdiv = flat_gp_div(flat_gp_const(6.0), flat_gp_const(2.0))
        self.assertEqual(self._ops(rdiv), self._ops(expect_rdiv))

    def test_chained_expressions(self):
        expr = (flat_gp_const(1.0) + flat_gp_const(2.0)) * flat_gp_const(3.0)
        expect = flat_gp_mul(
            flat_gp_add(flat_gp_const(1.0), flat_gp_const(2.0)), flat_gp_const(3.0)
        )
        self.assertEqual(self._ops(expr), self._ops(expect))

    def test_mixed_with_terminals(self):
        # combine terminals and consts
        expr = 0.0 - flat_gp_travel_time()
        expect = flat_gp_sub(flat_gp_const(0.0), flat_gp_travel_time())
        self.assertEqual(self._ops(expr), self._ops(expect))

        expr2 = flat_gp_time_until_due() * 2.0
        expect2 = flat_gp_mul(flat_gp_time_until_due(), flat_gp_const(2.0))
        self.assertEqual(self._ops(expr2), self._ops(expect2))


if __name__ == "__main__":
    unittest.main()
