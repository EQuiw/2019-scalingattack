import enum

class AreaNormEnumType(enum.Enum):
    """
    Specifies which norm will be used for the attack & defense evaluation for Area scaling.
    L0, L1, or L2 norm.
    """
    L0 = 0
    L1 = 1
    L2 = 2