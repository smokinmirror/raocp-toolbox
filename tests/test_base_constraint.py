import unittest
import numpy as np
import raocp.core.constraints.base_constraint as bc
import raocp.core.nodes as nodes


class TestBaseConstraint(unittest.TestCase):
    __state_size = 3
    __control_size = 2
    __nonleaf = nodes.Nonleaf()
    __leaf = nodes.Leaf()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_is_active_failure(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        with self.assertRaises(Exception):
            mock_con.is_active()

    def test_node_type_nonleaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        self.assertEqual(mock_con.node_type, TestBaseConstraint.__nonleaf)

    def test_node_type_leaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__leaf)
        self.assertEqual(mock_con.node_type, TestBaseConstraint.__leaf)

    def test_state_size_setter_nonleaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        mock_con.state_size = TestBaseConstraint.__state_size
        self.assertEqual(mock_con._Constraint__state_size, TestBaseConstraint.__state_size)

    def test_control_size_setter_nonleaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        mock_con.control_size = TestBaseConstraint.__control_size
        self.assertEqual(mock_con._Constraint__control_size, TestBaseConstraint.__control_size)

    def test_state_size_setter_leaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__leaf)
        mock_con.state_size = TestBaseConstraint.__state_size
        self.assertEqual(mock_con._Constraint__state_size, TestBaseConstraint.__state_size)
        self.assertEqual(mock_con.control_size, 0)

    def test_control_size_setter_leaf_failure(self):
        mock_con = bc.Constraint(TestBaseConstraint.__leaf)
        with self.assertRaises(Exception):
            mock_con.control_size = TestBaseConstraint.__control_size

    def test_state_matrix_setter_nonleaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        matrix = np.ones((TestBaseConstraint.__state_size, TestBaseConstraint.__state_size))
        mock_con.state_matrix = matrix
        self.assertTrue(np.array_equal(mock_con._Constraint__state_matrix, matrix))

    def test_control_matrix_setter_nonleaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        matrix = np.ones((TestBaseConstraint.__control_size, TestBaseConstraint.__control_size))
        mock_con.control_matrix = matrix
        self.assertTrue(np.array_equal(mock_con._Constraint__control_matrix, matrix))

    def test_state_matrix_setter_leaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__leaf)
        matrix = np.ones((TestBaseConstraint.__state_size, TestBaseConstraint.__state_size))
        mock_con.state_matrix = matrix
        self.assertTrue(np.array_equal(mock_con._Constraint__state_matrix, matrix))

    def test_control_matrix_setter_leaf_failure(self):
        mock_con = bc.Constraint(TestBaseConstraint.__leaf)
        matrix = np.ones((TestBaseConstraint.__control_size, TestBaseConstraint.__control_size))
        with self.assertRaises(Exception):
            mock_con.control_matrix = matrix

    def test_state_size_getter(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        mock_con._Constraint__state_size = TestBaseConstraint.__state_size
        self.assertEqual(mock_con.state_size, TestBaseConstraint.__state_size)

    def test_control_size_getter(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        mock_con._Constraint__control_size = TestBaseConstraint.__control_size
        self.assertEqual(mock_con.control_size, TestBaseConstraint.__control_size)

    def test_state_matrix_getter(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        self.assertEqual(mock_con.state_matrix, None)

    def test_control_matrix_getter(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        self.assertEqual(mock_con.control_matrix, None)

    def test_leaf_constraint(self):
        mock_con = bc.Constraint(TestBaseConstraint.__leaf)
        mock_con.state_size = TestBaseConstraint.__state_size
        self.assertEqual(mock_con._Constraint__control_size, 0)

    def test_state_transpose_nonleaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        matrix = np.ones((TestBaseConstraint.__state_size, TestBaseConstraint.__state_size + 2))
        mock_con.state_matrix = matrix
        mock_con.state_size = TestBaseConstraint.__state_size
        mock_con.control_size = TestBaseConstraint.__state_size
        self.assertTrue(np.array_equal(mock_con.state_matrix_transposed, matrix.T))

    def test_control_transpose_nonleaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__nonleaf)
        matrix = np.ones((TestBaseConstraint.__state_size, TestBaseConstraint.__state_size + 2))
        mock_con.control_matrix = matrix
        mock_con.control_size = TestBaseConstraint.__state_size
        mock_con.state_size = TestBaseConstraint.__state_size
        self.assertTrue(np.array_equal(mock_con.control_matrix_transposed, matrix.T))

    def test_state_transpose_leaf(self):
        mock_con = bc.Constraint(TestBaseConstraint.__leaf)
        matrix = np.ones((TestBaseConstraint.__state_size, TestBaseConstraint.__state_size + 2))
        mock_con.state_matrix = matrix
        mock_con.state_size = TestBaseConstraint.__state_size
        self.assertTrue(np.array_equal(mock_con.state_matrix_transposed, matrix.T))


if __name__ == '__main__':
    unittest.main()
