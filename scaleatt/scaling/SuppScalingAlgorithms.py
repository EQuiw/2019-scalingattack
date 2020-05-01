import enum

class SuppScalingAlgorithms(enum.Enum):
    """
    The supported scaling algorithms, so that we have one unique
    way to select NEAREST, or BILINEAR, CUBIC for the different
    scaling libraries.
    """
    NEAREST = 100
    LINEAR = 200
    CUBIC = 300
    LANCZOS = 400
    AREA = 500