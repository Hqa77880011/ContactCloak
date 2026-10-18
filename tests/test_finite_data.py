import unittest
import numpy as np
from finite_data import positions, sessions


class FiniteInputTests(unittest.TestCase):
    def test_kinematics_and_session_partition(self):
        np.testing.assert_allclose(positions(np.zeros(15)), [[0, 1, 0], [.3, 1, 0], [.6, 1, 0]])
        data = sessions(seed=17, frames=4)
        self.assertEqual(data["x"].shape, (24, 4, 15))
        self.assertEqual(data["anchors"].shape, (24, 4, 3, 3))
        roles = [set(data["group"][data["split"] == role]) for role in range(3)]
        self.assertFalse(roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
        for role in range(3):
            self.assertEqual(set(data["y"][data["split"] == role]), {0, 1, 2, 3})
        self.assertEqual(len(set(data["session_seed"])), 24)
