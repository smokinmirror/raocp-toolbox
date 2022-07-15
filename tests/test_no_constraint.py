import unittest
import raocp.core.nodes as core_nodes
import raocp.core.constraints.no_constraint as core_no_constraint


class TestNoConstraint(unittest.TestCase):
    __no_constraint_nl = None
    __no_constraint_l = None

    @staticmethod
    def _construct_mock_constraints():
        TestNoConstraint.__no_constraint = core_no_constraint.No()
        TestNoConstraint.__no_constraint_nl = core_no_constraint.No(core_nodes.Nonleaf())
        TestNoConstraint.__no_constraint_l = core_no_constraint.No(core_nodes.Leaf())

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestNoConstraint._construct_mock_constraints()

    def test_is_active(self):
        self.assertEqual(TestNoConstraint.__no_constraint.is_active, False)

    def test_is_active_nl(self):
        self.assertEqual(TestNoConstraint.__no_constraint_nl.is_active, False)

    def test_is_active_l(self):
        self.assertEqual(TestNoConstraint.__no_constraint_l.is_active, False)


if __name__ == '__main__':
    unittest.main()
