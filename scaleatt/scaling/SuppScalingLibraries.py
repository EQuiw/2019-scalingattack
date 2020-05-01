import enum

class SuppScalingLibraries(enum.Enum):
    """
    The scaling libraries that we currently support.
    OpenCV, Tensorflow, Pillow (PIL)
    """
    CV = 150
    TF = 250
    PIL = 350