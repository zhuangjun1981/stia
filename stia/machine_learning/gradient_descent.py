import matplotlib.pyplot as plt
import numpy as np
from stia.utility.image_analysis import array_nor_mean_range


class GradientDescent(object):
    """
    class for basic gradient descent linear regression

    attributes:

    self.x_arr: 2-d array, shape: number of samples x (number of features + 1), input feature, each column represents
                one feature (first column contains only 1s for intercept), each row represent one example, the
                data saved in this attribute are after scaling, each feature is subtracted by its mean and divided by
                its range, to regenerate original feature array, use self.get_original_feature_array() method
    self.x_mean: 2-d array, shape: 1 x (number of features + 1), mean of each feature, first element should be 1
    self.x_range: 2-d array, shape: 1 x (number of features + 1), range of each feature, first element should be 0
    self.y_arr: 2-d array, shape: number of samples x 1, target variable
    self.params: 2-d array, shape: number of iterations x (number of features + 1), linear coefficients for each
                 feature through iterations, elements in first row are intercepts
    self.errs: 1-d array, length: number of iterations, errors after each iteration
    self.end_iter: int, the number of last iteration
    self.iteration: int, the number of maximum iterations
    self.step: float, gradient descent update step
    self.reg_lambda: float, regularization coefficient
    """

    def __init__(self, iteration=1000, step=1., is_pass_origin=False, reg_lambda=None):
        """
        input matrices are set as None, please use .set_data() to input the training set.

        :param iteration: int, number of iterations
        :param step: float, gradient descent step
        :param is_pass_origin: bool, default: False. Does the model force to pass origin or not. if True, the first
                               number of parameters has no meaning.
        :param reg_lambda: float, lambda for simple regularization, if None, no regularization will be
               applied, when applied the update rule will have a regularization term, lambda * theta / m, theta is
               the parameter to fit, m is the total number of parameters
        """

        self.x_arr = None
        self.x_mean = None
        self.x_range = None
        self.y_arr = None
        self.params = None
        self.errs = None
        self.end_iter = None

        self.iteration = iteration
        self.step = step
        self.reg_lambda = reg_lambda
        self.is_pass_origin = is_pass_origin

    def __str__(self):

        string = '\n================================================================'
        string += '\nstia.machine_learning.gradient_descent.GradientDescent instance'
        string += '\nself.iteration:\t\t' + str(self.iteration)
        string += '\nself.step:\t\t\t' + str(self.step)
        string += '\nself.reg_lambda:\t' + str(self.reg_lambda)
        string += '\n------------------------------------------'

        if self.x_arr is None:
            string += '\nself.x_arr:\t\t\tNone'
        else:
            string += '\nself.x_arr.shape:\t' + str(self.x_arr.shape)

        string += '\nself.x_mean:\t\t' + str(self.x_mean)
        string += '\nself.x_range:\t\t' + str(self.x_range)

        if self.y_arr is None:
            string += '\nself.y_arr:\t\t\tNone'
        else:
            string += '\nself.y_arr.shape:\t' + str(self.y_arr.shape)

        if self.params is None:
            string += '\nself.params:\t\tNone'
        else:
            string += '\nself.params.shape:\t' + str(self.params.shape)

        if self.errs is None:
            string += '\nself.err:\t\t\tNone'
        else:
            string += '\nself.err.shape:\t\t' + str(self.errs.shape)

        if self.end_iter is None:
            string += '\nself.end_iter:\t\tNone'
        else:
            string += '\nself.end_iter:\t\t' + str(self.end_iter)

        string += '\n================================================================\n'

        return string

    def set_data(self, input_x, input_y, start_params=None):
        """
        input training dataset

        :param input_x: 2-d array, input variable matrix, shape: n x m. n is number of samples. m is number of input
                        variables, if 1-d array, assume each number is a sample. when saved into the class, a
                        constant input variable with value ones will be added as the first column to self.mat_x
        :param input_y: 2-d array, output variable matrix, shape: n x 1. n is number of samples., if 1-d array, assume
                        each number is a sample
        :param start_params: 2-d array, start parameter set, shape: (m + 1) x 1. m is number of input variables. if None
                            start from all zeros. if 1-d array, assume first element is for the intercept and rest
                            elements are coefficients for each input variable.

        :return: None
        """

        if self.x_arr is not None:
            raise ValueError('self.x_arr is not None. Please clear the class before set data.')

        if self.y_arr is not None:
            raise ValueError('self.y_arr is not None. Please clear the class before set data.')

        if self.params is not None:
            raise ValueError('self.params is not None. Please clear the class before set data.')

        if self.errs is not None:
            raise ValueError('self.err is not None. Please clear the class before set data.')

        if len(input_x.shape) == 1:
            x_arr = np.array([input_x.astype(np.float64)]).transpose()
        elif len(input_x.shape) == 2:
            x_arr = input_x.astype(np.float64)
        else:
            raise ValueError('input mat_x should be either 1-d array or 2-d array.')

        x_arr, x_range, x_mean = array_nor_mean_range(x_arr)
        if self.is_pass_origin:
            self.x_arr = np.hstack((np.zeros((x_arr.shape[0], 1)), x_arr))
            self.x_range = np.hstack(([[0.]], x_range))
            self.x_mean = np.hstack(([[0.]], x_mean))
        else:
            self.x_arr = np.hstack((np.ones((x_arr.shape[0], 1)), x_arr))
            self.x_range = np.hstack(([[0.]], x_range))
            self.x_mean = np.hstack(([[1.]], x_mean))

        # self.x_arr, self.x_range, self.x_mean = array_nor_mean_range(x_arr)

        if len(input_y.shape) == 1:
            if len(input_y) == self.x_arr.shape[0]:
                self.y_arr = np.array([input_y.astype(np.float64)]).transpose()
            else:
                raise ValueError('number of elements in mat_y should match number of samples in mat_x.')
        elif len(input_y.shape) == 2:
            if input_y.shape[0] == self.x_arr.shape[0] and input_y.shape[1] == 1:
                self.y_arr = input_y.astype(np.float64)
            else:
                raise ValueError('number of rows (samples) in mat_y should match number of rows (samples) in mat_x, '
                                 'and number of columns in mat_y should be 1.')
        else:
            raise ValueError('number of rows (samples) in mat_y should match number of rows (samples) in mat_x, '
                             'and number of columns in mat_y should be 1.')

        self.params = np.zeros((self.x_arr.shape[1], self.iteration), dtype=np.float64)
        self.params[:] = np.nan
        if start_params is None:
            self.params[:, 0] = 0.
        elif len(start_params.shape) == 1:
            if len(start_params) == self.x_arr.shape[1]:
                self.params[:, 0] = start_params
            else:
                raise ValueError('number of elements in start_params should equal number of input variables + 1.')
        elif len(start_params.shape) == 2:
            if (start_params.shape[0] == self.x_arr.shape[1]) and (start_params.shape[1] == 1):
                self.params[:, 0:1] = start_params
            else:
                raise ValueError('number of rows in start_params should equal number of input variables + 1, and'
                                 'number of columns in start_params should equal 1.')
        else:
            raise ValueError('start_params should be None, 1-d array or 2-d array.')

        self.errs = np.zeros(self.iteration, dtype=np.float64)
        self.errs[:] = np.nan

    @staticmethod
    def _get_error(x_arr, params, y_arr):
        return np.mean(np.power((np.dot(x_arr, params).flatten() - y_arr.flatten()), 2))

    def get_original_feature_array(self):
        """
        :return: original unscaled feature matrix without intercept term,
        2-d array, number of samples x number of features
        """
        return ((self.x_arr * self.x_range) + self.x_mean)[:, 1:]

    def get_final_parameters(self):
        """
        :return: final parameters set on unscaled features, 1-d array, length: number of feature + 1, first element is
                 intercept
        """

        x_range = self.x_range[:, 1:]  # range only for features without intercept
        x_mean = self.x_mean[:, 1:]  # mean only for features without intercept

        intercept_s = self.params[0, self.end_iter]  # intercept for scaled features
        params_s = self.params[1:, self.end_iter]  # params for scaled features without intercept
        intercept = np.array([intercept_s - np.sum((params_s * x_mean) / x_range)])  # intercept for unscaled features
        params = (params_s / x_range).flatten()  # params for unscaled features without intercept
        return np.hstack((intercept, params))

    def solve_normal_equation(self):
        """
        solve the linear regression with calculating the normal equation

        :return: params, 1-d array, length number of features + 1, first element is the intercept
        """

        x_mat = self.get_original_feature_array()
        if self.is_pass_origin:
            x_mat = np.hstack((np.zeros((x_mat.shape[0], 1)), x_mat))
        else:
            x_mat = np.hstack((np.ones((x_mat.shape[0], 1)), x_mat))
        y_mat = self.y_arr

        return np.dot(np.dot(np.linalg.pinv(np.dot(x_mat.transpose(), x_mat)), x_mat.transpose()), y_mat).flatten()

    def clear(self):
        self.x_arr = None
        self.x_range = None
        self.x_mean = None
        self.y_arr = None
        self.params = None
        self.errs = None

    def plot_errs(self, plot_axis=None):
        if self.errs is None:
            print('Can not plot errors. Regression has not run yet.')
        else:
            if plot_axis is None:
                f = plt.figure(figsize=(10, 10))
                plot_axis = f.add_subplot(111)
            errs = self.errs[np.bitwise_not(np.isnan(self.errs))]
            plot_axis.plot(errs)
            plot_axis.set_xlabel('iteration')
            plot_axis.set_ylabel('error (mean squared)')
            plot_axis.set_yscale('log')

        return plot_axis

    def run(self):
        """
        run the gradient descent loop, and check convergence for each iteration. if mean square error gets
        no less than previous iteration. Stop the loop. If convergence not detected, it will run through
        all iterations defined by self.iteration.

        this method also set the last iteration number to the attribute self.end_iter

        :return: int, number of iterations ran before convergence; if no training dataset (self.mat_x,
                 self.mat_y) detected, return None.
        """

        if self.x_arr is None or self.y_arr is None:
            print('No training dataset found. Please load data by using self.set_data() method.')
            return None

        for curr_iter in range(self.iteration):

            # print 'testing'
            # print curr_iter
            # print self.params[:, curr_iter]
            # print self._get_error(self.x_arr, self.params[:, curr_iter], self.y_arr)

            self.errs[curr_iter] = self._get_error(self.x_arr, self.params[:, curr_iter], self.y_arr)

            if curr_iter > 0 and self.errs[curr_iter] >= self.errs[curr_iter - 1]:
                print('iteration: {0:d}; convergence end. Current error: {1:f}; last error: {2:f}'
                      .format(curr_iter, self.errs[curr_iter], self.errs[curr_iter-1]))
                self.end_iter = curr_iter
                return curr_iter

            updates = (np.dot(self.x_arr, self.params[:, curr_iter:curr_iter + 1]) - self.y_arr).flat

            new_params = np.zeros(self.x_arr.shape[1])
            for fea_ind in range(self.x_arr.shape[1]):

                if fea_ind == 0 or self.reg_lambda is None:
                    new_params[fea_ind] = self.params[fea_ind, curr_iter] - \
                                          self.step * np.mean((self.x_arr[:, fea_ind] * updates).flat)
                else:
                    new_params[fea_ind] = self.params[fea_ind, curr_iter] - \
                                          self.step * np.mean((self.x_arr[:, fea_ind] * updates).flat) - \
                                          self.step * self.reg_lambda * self.params[fea_ind, curr_iter] / \
                                          self.x_arr.shape[1]

            if curr_iter < self.iteration - 1:
                self.params[:, curr_iter + 1] = new_params

        self.end_iter = curr_iter
        return curr_iter


if __name__ == '__main__':

    # =======================================================================
    # gd = GradientDescent(iteration=100, step=0.05)
    # gd.set_data(np.arange(5), 3 + 2.1 * np.arange(5), np.array([0., 0.]))
    # print gd
    #
    # gd.run()
    #
    # print gd.params[:, -1]
    #
    # f = plt.figure(figsize=(15, 4))
    # ax1 = f.add_subplot(131)
    # ax1.plot(gd.errs)
    # ax2 = f.add_subplot(132)
    # ax2.plot(gd.params[0, :])
    # ax3 = f.add_subplot(133)
    # ax3.plot(gd.params[1, :])
    # plt.show()
    # =======================================================================

    # =======================================================================
    gd = GradientDescent(iteration=1000, step=0.1)
    print gd
    x_arr = np.array([
                      [1., 2.],
                      [3., 6.],
                      [4., 3.],
                      [7., 10.],
                      [-3, 4.],
                      ])

    theta = np.array([
                      [2.5],
                      [3.],
                      [5.4]
                      ])

    y_arr = np.dot(np.hstack((np.ones((x_arr.shape[0], 1)), x_arr)), theta)

    gd.set_data(x_arr, y_arr)
    print gd

    p1_start = 0.
    p1_end = 60.
    p1_step = 1.

    p2_start = 0.
    p2_end = 80.
    p2_step = 1.

    param1 = np.arange(p1_start, p1_end, p1_step)
    param2 = np.arange(p2_start, p2_end, p2_step)
    mat_err = np.zeros((len(param1), len(param2)))

    for i, p1 in enumerate(param1):
        for j, p2 in enumerate(param2):
            mat_err[i, j] = gd._get_error(gd.x_arr, np.array([0, p1, p2]).transpose(), gd.y_arr)

    last_iter = gd.run()
    print '\nlast paramter set:\n', gd.params[:, last_iter]
    print gd

    f = plt.figure(figsize=(11, 8))
    ax = f.add_subplot(111)
    fig = ax.imshow(mat_err)
    ax.set_xlabel('param2')
    ax.set_ylabel('param1')
    ax.set_axis_off()
    f.colorbar(fig)

    for i, params in enumerate(gd.params.transpose()[0:last_iter+1]):
        curr_p1 = params[1]
        curr_p2 = params[2]
        ind_p1 = (curr_p1 - p1_start) // p1_step
        ind_p2 = (curr_p2 - p2_start) // p2_step
        ax.text(ind_p1, ind_p2, str(i))

    f2 = plt.figure(figsize=(15, 4))
    ax1 = f2.add_subplot(131)
    gd.plot_errs(ax1)
    ax2 = f2.add_subplot(132)
    ax2.plot(gd.params[1, :last_iter + 1])
    ax3 = f2.add_subplot(133)
    ax3.plot(gd.params[2, :last_iter + 1])
    plt.show()

    print gd.get_final_parameters()
    print gd.solve_normal_equation()
    # =======================================================================


