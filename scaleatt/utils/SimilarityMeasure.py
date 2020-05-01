import enum


class SimilarityMeasure(enum.Enum):
    """
    Different ways to compare a pair of images,
    by using feature matching (SIFT, ORF or combination of both),
    by using Peak-Signal-to-Noise Ratio,
    TODO in future, we might also move other ways to compare images, such as the histogram and scattering method.
    """

    SIFT = 1
    # ORF = 2
    SIFT_ORF = 3
    PSNR = 4


