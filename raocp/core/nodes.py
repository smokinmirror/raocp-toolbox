

class Node:
    """
    Empty class for storing node type
    """
    @property
    def is_nonleaf(self):
        return False

    @property
    def is_leaf(self):
        return False


class Nonleaf(Node):
    """
    Empty class for nonleaf type node
    """
    @property
    def is_nonleaf(self):
        return True


class Leaf(Node):
    """
    Empty class for leaf type node
    """
    @property
    def is_leaf(self):
        return True
