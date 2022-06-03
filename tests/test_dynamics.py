import unittest
import numpy as np
import raocp.core.dynamics as core_dynamics


class TestDynamics(unittest.TestCase):
    __size = 3

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_dynamics(self):
        # construct good state and control dynamics
        state_dynamics = np.eye(TestDynamics.__size)
        control_dynamics = np.ones((TestDynamics.__size, TestDynamics.__size + 1))

        # test
        _ = core_dynamics.Dynamics(state_dynamics, control_dynamics)

    def test_dynamics_failure(self):
        # construct bad state and control dynamics (unequal rows)
        state_dynamics = np.eye(TestDynamics.__size + 1)
        control_dynamics = np.ones((TestDynamics.__size, TestDynamics.__size + 1))

        # test with error catch
        with self.assertRaises(ValueError):
            _ = core_dynamics.Dynamics(state_dynamics, control_dynamics)


if __name__ == '__main__':
    unittest.main()
