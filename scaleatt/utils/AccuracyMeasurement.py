import numpy as np
import enum


class TopKAccuracyMeasurement(enum.Enum):
    """
    When can we assume that the attack is successful? Denote by y_target the output of target image.
    And y_pred (=y_is) is the output prediction vector of DNN for downscaled attack image.
    1) If the highest entry in y_pred is under the top-5 from y_target?
    2) If the highest entry of y_target is under the top-5 of y_pred?
    3) If one of the top-5 of y_target is under the top-5 of y_pred?
    4) same as 3), but we remove entries that have a zero probability and thus might only be matched by accident.
    """
    topk_m1 = 1
    topk_m2 = 2
    topk_m3 = 3
    topk_m4 = 4


class AccuracyMeasurement:
    # used as class, just to make it similar to SimilarityMeasurementTool...

    @staticmethod
    def measure_top_k_accuracy(y_should: np.ndarray,
                               y_is: np.ndarray,
                               k:int=5,
                               method=TopKAccuracyMeasurement.topk_m4) -> np.ndarray:
        """
        Expects both y_true and y_pred to be one-hot encoded.
        :param y_should: y_true or what we expect;  should be one-hot encoded.
        :param y_is: y_pred or what we obtain;   should be one-hot encoded.
        :param k: k in top-k accuracy
        :param method: method of measuring, type of TopKAccuracyMeasurement
        :return: matches
        """

        if method == TopKAccuracyMeasurement.topk_m1:
            return AccuracyMeasurement._top_k_accuracy_onesided(y_true=y_is, y_pred=y_should, k=k)
        elif method == TopKAccuracyMeasurement.topk_m2:
            return AccuracyMeasurement._top_k_accuracy_onesided(y_true=y_should, y_pred=y_is, k=k)
        elif method == TopKAccuracyMeasurement.topk_m3:
            return AccuracyMeasurement._top_k_inter_accuracy(y_other=y_should, y_pred=y_is, k=k)
        elif method == TopKAccuracyMeasurement.topk_m4:
            return AccuracyMeasurement._top_k_inter_accuracy_filter(y_other=y_should, y_pred=y_is, k=k)
        else:
            raise NotImplementedError()


    @staticmethod
    def _top_k_inter_accuracy(y_other, y_pred, k=1):
        """
        Compares in both directions if both prediction vectors share at least one label among the k predictions
        with highest score.
        Makes sense if we want to compare two prediction vectors, if actually same objects have actually same output,
        maybe just in slightly different order != top_k_accuracy, which can be used to compare against real label.
        :return: array of matches, boolean values.
        """
        argsorted_y = np.argsort(y_pred)[:, -k:]
        argsorted_ytrue = np.argsort(y_other)[:, -k:]

        matches: list = [len(set(argsorted_y[i, :]).intersection(argsorted_ytrue[i, :])) > 0 for i in
                         range(argsorted_y.shape[0])]

        return np.array(matches)


    @staticmethod
    def _top_k_inter_accuracy_filter(y_other, y_pred, k=1):
        """
        Same as top_k_inter_accuracy, but with filtering
            = remove classes that have a near zero probability.
        By using this function, we could also avoid 2-4 false positives in some experiments
            (false positive here = succesful attack although not clearly).
        :return: array of matches, boolean values.
        """
        argsorted_y = np.argsort(y_pred)[:, -k:]
        argsorted_ytrue = np.argsort(y_other)[:, -k:]
        bound = 0.03 # randomly chosen, close to 0

        matches = []
        for i in range(argsorted_y.shape[0]):
            ax = argsorted_y[i, np.where(y_pred[i, argsorted_y[i, :]] > bound)[0]]
            bx = argsorted_ytrue[i, np.where(y_other[i, argsorted_ytrue[i, :]] > bound)[0]]
            matches.append(len(set(ax).intersection(bx)) > 0)

        return np.array(matches)


    @staticmethod
    def _top_k_accuracy_onesided(y_true, y_pred, k=1):
        """ Returns match element-wise.
        Expects both y_true and y_pred to be one-hot encoded.
        """
        argsorted_y = np.argsort(y_pred)[:, -k:]
        return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0)


# *** Helper Functions ***


def top_k_accuracy(y_true, y_pred, k=1):
    """From: https://github.com/chainer/chainer/issues/606

    Expects both y_true and y_pred to be one-hot encoded.
    """
    import warnings
    warnings.warn('Deprecated use of top_k_accuracy')
    argsorted_y = np.argsort(y_pred)[:, -k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()


def top_k_accuracy_detailed(y_true, y_pred, k=1):
    """ Returns match element-wise.
    Expects both y_true and y_pred to be one-hot encoded.
    """
    import warnings
    warnings.warn('Deprecated use of top_k_accuracy')
    argsorted_y = np.argsort(y_pred)[:, -k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0)


def top_k_inter_accuracy(y_other, y_pred, k=1):
    """
    Compares in both directions if both prediction vectors share at least one label among the k predictions
    with highest score.
    Makes sense if we want to compare two prediction vectors, if actually same objects have actually same output,
    maybe just in slightly different order != top_k_accuracy, which can be used to compare against real label.
    :return: array of matches, boolean values.
    """
    raise DeprecationWarning()
    # argsorted_y = np.argsort(y_pred)[:, -k:]
    # argsorted_ytrue = np.argsort(y_other)[:, -k:]
    #
    # matches: list = [len(set(argsorted_y[i,:]).intersection(argsorted_ytrue[i,:]))>0 for i in range(argsorted_y.shape[0])]
    #
    # return np.array(matches)



