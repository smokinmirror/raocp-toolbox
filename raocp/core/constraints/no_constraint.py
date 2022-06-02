import raocp.core.constraints.base_constraint as bc


class No(bc.Constraint):
    """
    For no constraints
    """
    def __init__(self, node_type=None):
        super().__init__(node_type)

    @property
    def is_active(self):
        return False
