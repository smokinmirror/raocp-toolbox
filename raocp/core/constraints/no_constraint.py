import raocp.core.constraints.base_constraint as bc


class No(bc.Constraint):
    """
    For no constraints
    """
    def __init__(self):
        super().__init__()

    @property
    def is_active(self):
        return False
