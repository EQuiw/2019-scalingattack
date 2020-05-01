import sys
import skimage.measure
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from utils.SimilarityMeasure import SimilarityMeasure


class SimilarityMeasurementTool:

    @staticmethod
    def sim_measure(img1: np.ndarray, img2: np.ndarray, similarity_measurement: SimilarityMeasure):
        if similarity_measurement == SimilarityMeasure.SIFT:
            simscore = SimilarityMeasurementTool.match_sift(img1=img1, img2=img2,
                                                         doprint=False,
                                                         threshold=0.6)
        elif similarity_measurement == SimilarityMeasure.SIFT_ORF:
            simscore1 = SimilarityMeasurementTool.match_sift(img1=img1, img2=img2,
                                                          doprint=False, threshold=0.6)
            simscore2 = SimilarityMeasurementTool.match_orb(img1=img1, img2=img2,
                                                         doprint=False, threshold=0.6)
            simscore = (simscore1 + simscore2) / 2

        elif similarity_measurement == SimilarityMeasure.PSNR:
            simscore = skimage.measure.compare_psnr(img1, img2)
        else:
            raise Exception("Unknown argument for similarity measurement")

        return simscore



    @staticmethod
    def match_orb(img1, img2, doprint: bool = False, threshold: float = .75):
        """
        Feature matching with ORB
        :param img1: image under investigation 1
        :param img2: image under investigation 2
        :param doprint: verbose flag
        :param threshold: float value as threshold
        :return:
        """
        # Initiate detector
        try:
            orb = cv.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            # BFMatcher with default params
            bf = cv.BFMatcher(cv.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply ratio test
            good = []

            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good.append([m])
            # print(len(matches))
            # print(len(good))
            # cv.drawMatchesKnn expects list of lists as matches.
            if doprint is True:
                img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(img3), plt.show()
            return len(good)
        except:
            print("Problem with current matching; Return -1", file=sys.stderr)
            return -1.0


    @staticmethod
    def match_sift(img1, img2, doprint: bool = False, threshold: float = .75):
        """
        Feature matching with SIFT.
        Important: Usage of SIFT requires compilation of OpenCV, and consider the license as well!
        If OpenCV was not installed with SIFT from source, this method will not work!
        :param img1: image under investigation 1
        :param img2: image under investigation 2
        :param doprint: verbose flag
        :param threshold: float value as threshold
        :return:
        """
        # Initiate SIFT detector

        sift = cv.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2)
        if doprint is True:
            img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3), plt.show()
        return len(good)
