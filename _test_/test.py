"""
Test the optimization module.

Yujia Li, 09/2014
"""

import pyopt.opt as opt
import numpy as np

_OPT_CHECK_EPS = 1e-3

def vec_str(v):
    s = '[ '
    for i in range(len(v)):
        s += '%11.8f ' % v[i]
    s += ']'
    return s

def test_vec_pair(v1, msg1, v2, msg2):
    print msg1 + ' : ' + vec_str(v1)
    print msg2 + ' : ' + vec_str(v2)
    n_space = len(msg2) - len('diff')
    print ' ' * n_space + 'diff' + ' : ' + vec_str(v1 - v2)
    err = np.sqrt(((v1 - v2)**2).sum())
    print 'err : %.8f' % err

    success = err < _OPT_CHECK_EPS
    print '** SUCCESS **' if success else '** FAIL **'

    return success

def test_gradient_descent(weight_decay=0):
    print 'Testing gradient descent weight_decay=%g' % weight_decay

    def f(x):
        return ((x-1)**2).sum()/2.0, x-1

    def f_info(x):
        return 'x=%s' % str(x)

    x0 = np.array([2, 0], dtype=np.float)
    x_opt = opt.fmin_gradient_descent(f, x0, learn_rate=0.6, momentum=0.0, 
        weight_decay=weight_decay, learn_rate_drop_iters=5, adagrad_start_iter=1,
        max_iters=20, iprint=1, f_info=f_info, verbose=True)

    x_star = np.ones(2, dtype=np.float) / (1 + weight_decay)

    success = test_vec_pair(x_star, 'x_star', x_opt, 'x_opt ')
    print ''
    return success

def test_all_gradient_descent():
    print '========================'
    print 'Testing Gradient Descent'
    print '========================'

    n_success = 0

    if test_gradient_descent(0):
        n_success += 1
    if test_gradient_descent(1):
        n_success += 1

    n_total = 2
    return n_success, n_total

def run_all_tests():
    test_list = [test_all_gradient_descent]

    n_total = 0
    n_success = 0
    for test in test_list:
        success, total = test()
        n_success += success
        n_total += total

    print ''
    print '==========================='
    print 'All tests finished. Summary: %d/%d success, %d failed' % (n_success, n_total, n_total - n_success)
    print ''

if __name__ == '__main__':
    run_all_tests()
