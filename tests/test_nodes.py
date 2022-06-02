import unittest
import raocp.core.nodes as nodes


class TestNodes(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_nonleaf(self):
        nonleaf_node = nodes.Nonleaf()
        self.assertTrue(nonleaf_node.is_nonleaf)
        self.assertFalse(nonleaf_node.is_leaf)

    def test_leaf(self):
        leaf_node = nodes.Leaf()
        self.assertFalse(leaf_node.is_nonleaf)
        self.assertTrue(leaf_node.is_leaf)


if __name__ == '__main__':
    unittest.main()
