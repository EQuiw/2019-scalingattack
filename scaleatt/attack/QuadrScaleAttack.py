from attack.ScaleAttackStrategy import ScaleAttackStrategy
# from attack.scale_coefficient_recovery import scale_cv2
import numpy as np
import cvxpy as cp
import typing
# import cv2 as cv


from scaling.ScalingApproach import ScalingApproach


class QuadraticScaleAttack(ScaleAttackStrategy):
    """
    Implements an image-scaling attack by solving it as optimization problem.
    """

    def __init__(self, eps: typing.Union[float, typing.List[float]], verbose: bool):
        super().__init__(verbose)

        # for backwards compability, we can pass an eps value as float or as list.
        if type(eps) != list:
            eps = [eps]
        self.eps: typing.List[float] = eps
        self.round_to_integer = True
        self.boundRight = 255

        # This is an optimization possibility. If true, we exclude all dimensions that are zero and thus
        #   do not contribute to scaling (bee below). If scaling matrix is very sparse, we can save time.
        self.optimize_runtime: bool = False

    # @Overwrite
    def _attack_ononedimension(self, src_image: np.ndarray, target_image: np.ndarray,
               scaler_approach: ScalingApproach) \
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:


        # w.r.t Algorithm 2 in paper Xiao et al.
        # A. Step 1 (lines 1-8)
        scaledsrc1 = scaler_approach.scale_image_with(xin=src_image, trows=src_image.shape[0],
                               tcols=target_image.shape[1])

        attackimg1, optvalues1 = self._attack_in_one_direction(
            src_image=scaledsrc1, target_image=target_image, cl_matrix=scaler_approach.cl_matrix
        )

        if self.round_to_integer is True:
            attackimg1 = np.clip(np.round(attackimg1), 0, 255)

        # return attackImg1, optValues1, optValues1
        # B. Step 2 (lines 9-14)
        attackimg2, optvalues2 = self._attack_in_one_direction(
            src_image=src_image.T, target_image=attackimg1.T, cl_matrix=scaler_approach.cr_matrix.T
        )


        if self.round_to_integer is True:
            attackimg2 = np.clip(np.round(attackimg2), 0, 255)

        return attackimg2.T, optvalues1, optvalues2


    def _attack_in_one_direction(self, src_image: np.ndarray,
                                  target_image: np.ndarray,
                                  cl_matrix: np.ndarray):
        """
        Performs the attack in one direction (horizontal). If vertical needed, transpose the matrices first.
        """

        if self.optimize_runtime is False:
            attack_image_all = np.zeros(src_image.shape)
            useonly = np.arange(src_image.shape[0])
        else:
            # if true, we only choose considered pixels. Columns without impact are ignored.
            attack_image_all = np.copy(src_image).astype(np.float64) # np.zeros also creates float dtype.
            useonly = np.where(np.sum(cl_matrix, axis=0))[0]

        opt_values = np.zeros(src_image.shape[1])

        # for debug. purposes; we finally show a msg once if we had to use a higher eps. value at some point.
        highest_eps = self.eps[0]

        # go horizontal
        for h in range(src_image.shape[1]):

            if self.verbose is True and (h % 20) == 0:
                print("Iteration: {}".format(h))

            src_image_h = src_image[useonly, h]
            target_image_h = target_image[:, h]

            # we try various eps values, and if we succeed, we break; otherwise, if problem cannot be solved
            #   with current epsilon, we try another one. If we cannot solve the problem with any eps., we stop.
            opti_prob, opti_delta = None, None
            for cureps in self.eps:
                delta1 = cp.Variable(src_image_h.shape[0])
                ident = np.identity(src_image_h.shape[0])
                obj = (1 / 2) * cp.quad_form(delta1, ident)
                att_img = (src_image_h + delta1)
                constr1 = att_img <= self.boundRight
                constr2 = att_img >= 0

                cl_matrix_view = cl_matrix[:, useonly]
                constr3 = cp.abs(cl_matrix_view @ att_img - target_image_h) <= cureps

                prob = cp.Problem(cp.Minimize(obj), [constr1, constr2, constr3])

                prob_indicator: bool = self.__solve_problem(prob=prob, cureps=cureps, h=h)
                if prob_indicator is True:
                    opti_prob = prob
                    opti_delta = delta1
                    if cureps > highest_eps:
                        highest_eps = cureps
                    break
                else:
                    # print("eps did not work out for {} with {}".format(h, cureps))
                    continue

            if opti_prob is None:
                print("Could not solve at {} when creating the attack image.".format(h))
                # attack_image_all[:, h] = src_image_h
                raise Exception("Could not solve at {} -- attack image generation".format(h))

            opt_values[h] = opti_prob.value

            assert opti_delta.value is not None
            attack_image_all[useonly, h] = src_image_h + opti_delta.value


        if highest_eps > self.eps[0]:
            print("Had to use another eps than the first eps value: {}".format(highest_eps))

        return attack_image_all, opt_values


    def __solve_problem(self, prob, cureps, h) -> bool:
        """
        Solve problem, first with default solver, and then try ECOS,
        as trying multiple solvers is recommended by cvxpy docu.
        :return: true if problem was solved, false if not (error or not feasible => not optimal).
        """
        try:
            prob.solve()
        except:
            if self.verbose is True:
                print("QSQP failed")
            try:
                prob.solve(solver=cp.ECOS)
            except:
                if self.verbose is True:
                    print("Could not solve with QSPS and ECOS with {} at {}".format(cureps, h))
                return False

        if prob.status != cp.OPTIMAL and prob.status != cp.OPTIMAL_INACCURATE:
            if self.verbose is True:
                print("Could only solve with {} at {} with status: {}".format(cureps, h, prob.status))
            return False
        return True
