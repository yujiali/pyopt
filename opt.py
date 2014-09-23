"""
A set of functions to minimize an objective function, tailored to my pynn
package.

Other optimization packages can be found in scipy.optimize.

Yujia Li, 09/2014
"""

import math

class ParamSchedule(object):
    """
    Gradually changing schedule for a single variable.
    """
    def __init__(self, x_init, x_schedule_dict=None, drop_after_iters=10, 
            freeze_after_iters=-1, decrease_type='linear'):
        """
        x: initial variable value
        x_schedule_dict: a dict of (#iter -> var_value) the user defined schedule
            will be used instead of a gradually decreasing schedule
        drop_after_iters: decrease variable according to a 1/t schedule after every
            drop_iters iterations, if set to 0 the variable will not be decreased.
        freeze_after_iters: freeze variable after freeze_after_iters iterations if 
            freeze_after_iters >= 0.
        decrease_type: 'linear' or 'sqrt'
        """
        self.x_init = x_init
        self.schedule_dict = x_schedule_dict
        self.drop_after_iters = drop_after_iters
        self.freeze_after_iters = freeze_after_iters
        self.decrease_type = decrease_type

        if self.freeze_after_iters >= 0:
            self.x_freeze = self._get_var_at_iter(self.freeze_after_iters)

    def _get_var_at_iter(self, n_iter):
        """
        Return the variable value at iteration number n_iter without considering
        the variable freeze constraint.
        """
        if self.schedule_dict is not None:
            # user specified schedule
            schedule = sorted(self.schedule_dict.items(), key=lambda x: x[0], reverse=True)
            for k, v in schedule:
                if n_iter >= k:
                    return v
            return self.x_init
        else:   # gradually decreasing schedule
            if self.drop_after_iters > 0:
                factor = 1 + (n_iter - 1) / self.drop_after_iters
            else:
                factor = 1

            if self.decrease_type == 'sqrt':
                return 1.0 * self.x_init / math.sqrt(factor)
            else:
                return 1.0 * self.x_init / factor

    def get_var_at_iter(self, n_iter):
        """
        Return the variable value at iteration number n_iter.
        """
        if self.freeze_after_iters >= 0 and n_iter >= self.freeze_after_iters:
            return self.x_freeze
        else:
            return self._get_var_at_iter(n_iter)

class LearningSchedule(object):
    """
    This class deals with different learning rate, momentum schedules.
    """
    def __init__(self, learn_rate=1e-2, momentum=0, learn_rate_schedule=None,
            momentum_schedule=None, learn_rate_drop_iters=10,
            adagrad_start_iter=-1, decrease_type='linear'):
        self.learn_rate_schedule = ParamSchedule(learn_rate, 
                x_schedule_dict=learn_rate_schedule, 
                drop_after_iters=learn_rate_drop_iters,
                freeze_after_iters=adagrad_start_iter, 
                decrease_type=decrease_type)

        self.momentum_schedule = ParamSchedule(momentum,
                x_schedule_dict=momentum_schedule)

    def get_learn_rate_and_momentum(self, n_iter):
        """
        Return the proper learning rate and momentum to use for iteration
        number n_iter.
        """
        return self.learn_rate_schedule.get_var_at_iter(n_iter), \
                self.momentum_schedule.get_var_at_iter(n_iter)

def fmin_gradient_descent(f, x0, fprime=None, learn_rate=1e-2, momentum=0, 
        weight_decay=0, learn_rate_drop_iters=10, adagrad_start_iter=-1,
        max_iters=1000, iprint=1, f_info=None, verbose=True):
    """
    Minimize function f using gradient descent.

    f: the function to be minimized, accept a numpy vector as input and return
        an objective value. If fprime=None, this should return a tuple of 
        objective and gradient.
    fprime: fprime(x) computes the gradient of f at x. If None, gradient should
        be computed within f.
    x0: initial value of x
    learn_rate: learning rate, can be a single number (subject to the automatic
        schedule), or a dictionary of (#iter -> learn_rate), which is a user
        defined schedule
    momentum: momentum, can be a single number (fixed momentum) or a dictionary
        of (#iter -> momentum), which is a user defined schedule
    weight_decay: if > 0, a L2 term 1/2*weight_decay*x^2 will be added to 
        objective, gradients will be changed accordingly.
    learn_rate_drop_iters: decrease learning rate according to 1/t after every
        learn_rate_drop_iters iterations.
    adagrad_start_iter: the iteration to start using AdaGrad.
    max_iters: maximum number of iterations
    iprint: print optimization information every iprint iterations. Nothing 
        will be printed if iprint <= 0
    f_info: f_info(x) returns a string, which may be useful for monitoring the 
        optimization process
    verbose: print optimization information if True

    Return: x_opt, the x found after max_iters iterations.
    """
    if fprime is not None:
        f_and_fprime = lambda x: (f(x), fprime(x))
    else:
        f_and_fprime = lambda x: f(x)



def fmin_sgd():
    """
    Stochastic gradient descent optimization.
    """
    pass
