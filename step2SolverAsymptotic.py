# Asymptotic step 2 solver
# The solver solves for the dual problem and finds the lower bound for key rate

# Idea behind this code
#
#       The basic mathematical framework is based on arXiv:1710.05511
#   (https://arxiv.org/abs/1710.05511).
#
#    We use the same notation as in the Appendix D, lemma 12.
#
#   Note we implement a variant of the lemma 12 to take into account inequality constratins
#
#
# Syntax
#     [lowerbound, flag] = step2Solver(rho, Gamma, gamma, Gamma_ineq, gamma_ineq keyMap, krausOp, options)
#
# Input:
#
# *   rho - a density matrix rho_AB from step 1 calculation
#
# *   Gamma - a cell of Hermitian operators corresponding to equality
# constraints.
#
# *   gamma - a list of expectation values corresponding to observed statistics
#
# *   Gamma_ineq - a cell of Hermitian operators corresponding to
# inequality constraints.
#
# *   gamma_ineq - a list of expectation values corresponding to upper
# bound for the inequality constraints
#
# *   keyMap - Alice's key-generating PVM
#
# *   krausOp- a cell of Kraus operators corresponding to
# post-selection
#
# *   options - a structure that contains options for optimization
#
# Outputs:
#
# *  lowerbound - the lower bound of "key rate" (without error correction term, without rescaling back to log2)
#
# *  flag - a flag to indicate whether the optimization problem is solved
# successfully and whether we can trust the numerical result in the variable lowerbound

import numpy as np
import cvxpy as cp

from default_options import get_default_opt
from help_functions import primalfep, primalDfep, krausFunc



def sdpCondition(dualY, GammaVector):
    k = dualY.size 
    result = np.zeros((k,k))
    for iConstraint in range(k):
        result = result +  (dualY[iConstraint] * GammaVector[iConstraint])
    return result


def submaxproblem(GammaVector, gammaVector, gradfTranspose):
    nConstraints = len(gammaVector)
    totalDim = np.shape(gradfTranspose)[0]

    dualY = cp.Variable((nConstraints))
    CONSTR = gradfTranspose - sdpCondition(dualY,GammaVector)
    constraints = [
            CONSTR == cp.Variable(shape=(totalDim, totalDim)),
        -dualY >= 0
    ]

    problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(gammaVector , dualY))) , constraints)
    problem.solve(solver='CVXOPT',verbose = True)

    if problem.status == 'Infeasible':
        print("**** Warning: step 2 solver exception, submaxproblem status: %s ****\n", problem.status)

    return dualY.value, str(problem.status)


def step2SolverAsymptotic(rho, Gamma, gamma, Gamma_ineq, gamma_ineq, keyMap, krausOp, options):

    default_options = get_default_opt('Asymptotic2')

    for option_name in default_options.option_names:
        if not hasattr(options, option_name):
            setattr(options, option_name, getattr(default_options, option_name))
            print(f"**** solver 2 using default {option_name}: {getattr(default_options, option_name)} ****")

    epsilonprime = options.epsilonprime
    fval, epsilon1 = primalfep(rho, keyMap, krausOp)
    gradf, epsilon2 = primalDfep(rho, keyMap, krausOp)
    fval = np.real(fval)

    if  len(krausOp)==0:
        dprime = np.size(rho, 1)
    else:
        dprime = np.size(krausFunc(rho, krausOp), 1)

    epsilon = max(epsilon1, epsilon2)
    if epsilon > 1 / (np.exp(1) * (dprime - 1)):
        raise Exception('step2Solver:epsilon too large',
                        'Theorem cannot be applied. Please have a better rho to start with')

    Lepsilon = np.real(fval - np.trace(np.dot(rho, gradf)))

    nConstraints = len(Gamma)
    gamma_eps = np.array(gamma,dtype=float)+epsilonprime
    gamma_eps_min = -np.array(gamma,dtype=float)+epsilonprime
    if len(gamma_ineq)==0:
        gammaVector = np.hstack((gamma_eps,gamma_eps_min))
    else:
        gammaVector = np.concatenate(gamma + epsilonprime,-gamma + epsilonprime,np.array(gamma_ineq))
    minusGamma = []

    for i in range(nConstraints):
        minusGamma.append(-Gamma[i]) 

    GammaVector = np.concatenate([Gamma,minusGamma])

    dualY, status = submaxproblem(GammaVector, gammaVector, gradf)
    #Mepsilonprime = np.sum(np.dot(gammaVector, dualY))

    if epsilon == 0:
        zetaEp = 0
    else:
        zetaEp = 2 * epsilon * (dprime - 1) * np.log(dprime / (epsilon * (dprime - 1)))

    lowerbound = Lepsilon  - zetaEp #+ np.real(Mepsilonprime)

    return lowerbound, status
