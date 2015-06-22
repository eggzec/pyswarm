from __future__ import absolute_import, division, print_function
import unittest
from pyswarm.pso import pso


def banana_surface(point):
    x, y = point
    return x**4 - 2*y*x**2 + y**2 + x**2 - 2*x + 5


def constrain_banana(point):
    x, y = point
    return [-(x + 0.25)**2 + 0.75*y]


class TestPSO(unittest.TestCase):

    def setUp(self):
        self.low_bound = [-3, -1]
        self.up_bound = [2, 6]

    def test_pso_banana(self):
        x, f = pso(banana_surface, self.low_bound, self.up_bound, maxiter=1000,
                   swarmsize=50)
        x, y = x
        self.assertAlmostEqual(x, 1, places=2)
        self.assertAlmostEqual(y, 1, places=2)
        self.assertAlmostEqual(f, 4, places=3)

    def test_pso_banana_constrained(self):
        x, f = pso(banana_surface, self.low_bound, self.up_bound,
                   f_ieqcons=constrain_banana, maxiter=10000, swarmsize=50)
        x, y = x
        self.assertTrue(0.43 < x < 0.52)
        self.assertTrue(0.68 < y < 0.85)

    def test_examples_work(self):
        import pso_examples


if __name__ == '__main__':
    pass
    