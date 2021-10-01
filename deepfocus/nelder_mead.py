"""
LICENSE GPLv2: https://github.com/fchollet/nelder-mead/blob/c3022410201c9e09d9fbb091c2f458c7126ec610/LICENSE
Adopted from https://github.com/fchollet/nelder-mead/blob/master/nelder_mead.py
Pure Python/Numpy implementation of the Nelder-Mead algorithm.
Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
PS 2021: Algorithmic changes were made to the reduction step and by adding an optional noise term every 5 iterations.
"""

import copy
import numpy as np


def nelder_mead(f, x_start, step, no_improve_thr=10e-3, pred_details: dict = None,
                no_improv_break=10, max_iter=0, score_tol: float = None, jitter: tuple = None,
                alpha=1., gamma=2., rho=0.5, sigma=0.5, track_progress: bool = False):
    """

    Args:
        f: function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        x_start: initial position
        step: look-around radius in initial step
        no_improve_thr: break after no_improv_break iterations with
            an improvement lower than no_improv_thr.
        no_improv_break: break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        max_iter: always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        alpha: Parameters of the algorithm (see Wikipedia page for reference).
        gamma: Parameters of the algorithm (see Wikipedia page for reference).
        rho: Parameters of the algorithm (see Wikipedia page for reference).
        sigma: Parameters of the algorithm (see Wikipedia page for reference).
        score_tol: Optional, absolute score threshold
        track_progress: Return all intermediate results.
        jitter: Value range (absolut value) used to draw noise uniformly (-jitter, +jitter) for altering
            simplices in case the optimization gets stuck (no improvement for more than 5 iterations).
        pred_details: Dictionary in which the function call count is stored. Requires key "func_call_cnt".


    Returns:
        best parameter array, best score, parameter s.d. of last vertices), iters.
    """
    # init
    dim = len(x_start)
    prev_best = f(x_start)
    if pred_details is not None:
        pred_details['func_call_cnt'] += 1
    no_improv = 0
    res = [[x_start, prev_best]]
    overall_res = []

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step[i]
        score = f(x)
        if pred_details is not None:
            pred_details['func_call_cnt'] += 1
        res.append([x, score])

    # simplex iter
    iters = 0
    last_jitter = 0
    while 1:
        # -- add noise in case progress gets stuck
        if jitter is not None and no_improv > 5 and (iters - last_jitter) > 5:
            nres = [res[0], ]
            for tup in res[1:]:
                jitter_ = np.random.uniform(-np.array(jitter), np.array(jitter))
                print('Adding noise to non-best simplex:', jitter_)
                jit_x = np.array(tup[0]) + jitter_
                score = f(jit_x)
                if pred_details is not None:
                    pred_details['func_call_cnt'] += 1
                nres.append([jit_x, score])
            res = nres
            last_jitter = iters
        # --

        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]
        print(best)
        if track_progress:
            # track current best parameter values, score, s.d. of parameter vertices
            overall_res.append((*copy.copy(res[0]), np.std([x[0] for x in res], axis=0)))
        # break after max_iter
        if max_iter and iters >= max_iter:
            print('Max iter')
            if track_progress:
                return overall_res
            return res[0] + [np.std([x[0] for x in res], axis=0)], iters
        iters += 1

        # break after no_improv_break iterations with no improvement
        if score_tol is not None and score_tol >= res[0][1]:
            print('Reached absolute score threshold.')
            if track_progress:
                return overall_res
            return res[0] + [np.std([x[0] for x in res], axis=0)], iters

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            print('No improvement')
            if track_progress:
                return overall_res
            return res[0] + [np.std([x[0] for x in res], axis=0)], iters

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if pred_details is not None:
            pred_details['func_call_cnt'] += 1
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if pred_details is not None:
                pred_details['func_call_cnt'] += 1
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if pred_details is not None:
            pred_details['func_call_cnt'] += 1
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = [res[0]]
        for tup in res[1:]:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            if pred_details is not None:
                pred_details['func_call_cnt'] += 1
            nres.append([redx, score])
        res = nres

