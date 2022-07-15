import unittest
import numpy as np
import raocp.core.nodes as nodes
import raocp.core.constraints.rectangle as bc


class TestRectangle(unittest.TestCase):
    __state_size = 3
    __control_size = 2
    __min = 4 * np.ones((__state_size + __control_size, 1))
    __max = 5 * np.ones((__state_size + __control_size, 1))
    __nonleaf = nodes.Nonleaf()
    __leaf = nodes.Leaf()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_is_active(self):
        mock_con = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, TestRectangle.__max)
        self.assertEqual(mock_con.is_active, True)

    def test_check_constraints(self):
        good_max = np.asarray([None] * TestRectangle.__max.size)
        _ = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, good_max)

    def test_check_constraints_failure(self):
        bad_max = np.delete(TestRectangle.__max, 0)
        _none = np.asarray([None] * (TestRectangle.__state_size + TestRectangle.__control_size)).reshape(-1, 1)
        with self.assertRaises(Exception):
            _ = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, bad_max)
        with self.assertRaises(Exception):
            _ = bc.Rectangle(TestRectangle.__nonleaf, _none, _none)
        with self.assertRaises(Exception):
            _ = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__max, TestRectangle.__min)

    def test_state_size_and_matrix_setter(self):
        mock_con = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, TestRectangle.__max)
        mock_con.control_size = TestRectangle.__control_size
        mock_con.state_size = TestRectangle.__state_size
        self.assertEqual(mock_con.state_size, TestRectangle.__state_size)
        state_matrix = np.vstack((np.eye(TestRectangle.__state_size),
                                  np.zeros((TestRectangle.__control_size, TestRectangle.__state_size))))
        self.assertTrue(np.array_equal(mock_con.state_matrix, state_matrix))

    def test_control_size_and_matrix_setter(self):
        mock_con = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, TestRectangle.__max)
        mock_con.state_size = TestRectangle.__state_size
        mock_con.control_size = TestRectangle.__control_size
        self.assertEqual(mock_con.control_size, TestRectangle.__control_size)
        control_matrix = np.vstack((np.zeros((TestRectangle.__state_size, TestRectangle.__control_size)),
                          np.eye(TestRectangle.__control_size)))
        self.assertTrue(np.array_equal(mock_con.control_matrix, control_matrix))

    def test_check_input(self):
        mock_con = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, TestRectangle.__max)
        num = 10
        mock_con._Constraint__state_matrix = np.zeros((num, num))
        vector = np.zeros(num)
        mock_con._check_input(vector)

    def test_check_input_failure(self):
        mock_con = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, TestRectangle.__max)
        num = 10
        mock_con._Constraint__state_matrix = np.zeros((num, num))
        vector = np.zeros(num + 1)
        with self.assertRaises(Exception):
            mock_con._check_input(vector)

    def test_constrain(self):
        mock_con = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, TestRectangle.__max)
        num = TestRectangle.__state_size + TestRectangle.__control_size
        mock_con.state_size = TestRectangle.__state_size
        mock_con.control_size = TestRectangle.__control_size
        vector = 10 * np.asarray([np.random.randn(1) for _ in range(num)])
        for i in range(num):
            con_vec = mock_con._constrain(vector[i], TestRectangle.__min[i], TestRectangle.__max[i])
            self.assertTrue(TestRectangle.__min[i] <= con_vec <= TestRectangle.__max[i])

    def test_constrain_failure(self):
        mock_con = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, TestRectangle.__max)
        num = TestRectangle.__state_size
        mock_con.state_size = num
        vector = np.asarray([np.nan] * 3)
        for i in range(num):
            with self.assertRaises(ValueError):
                _ = mock_con._constrain(vector[i], TestRectangle.__min[i], TestRectangle.__max[i])

    def test_project(self):
        mock_con = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, TestRectangle.__max)
        num = TestRectangle.__state_size + TestRectangle.__control_size
        mock_con.state_size = TestRectangle.__state_size
        mock_con.control_size = TestRectangle.__control_size
        vector = 10 * np.asarray([np.random.randn(1) for _ in range(num)])
        con_vec = mock_con.project(vector)
        for i in range(num):
            self.assertTrue(TestRectangle.__min[i] <= con_vec[i] <= TestRectangle.__max[i])

    def test_leaf_constraint(self):
        mock_con = bc.Rectangle(TestRectangle.__nonleaf, TestRectangle.__min, TestRectangle.__max)
        num = TestRectangle.__state_size
        mock_con.state_size = num
        vector = 10 * np.asarray([np.random.randn(1) for _ in range(num)])
        for i in range(num):
            con_vec = mock_con._constrain(vector[i], TestRectangle.__min[i], TestRectangle.__max[i])
            self.assertTrue(TestRectangle.__min[i] <= con_vec <= TestRectangle.__max[i])


if __name__ == '__main__':
    unittest.main()
