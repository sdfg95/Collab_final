# FUNCTION NAME: step1SolverAsymptotic
# Asymptotic step 1 solver
# The solver first finds a rho closest to rho0 within the feasible region
# It then performs multiple Frank-Wolfe iterations
# (which uses the gradient to linearizes the objective function)
# to iteratively approach the local minimum.
#
# Input: initial rho0, keyMap, the observable-expectation pairs, options
#
# Output: optimal rho, optimal primalf value fval, the gap from the last
# iteration, and success flag.
from default_options import get_default_opt
from help_functions import removeLinearDependence, primalDf, primalf
import numpy as np
import cvxpy as cp
import time
from scipy.optimize import minimize_scalar

#%%%%%
def closestDensityMatrix(rho0, observables, expectations, options, status):
    expectations = expectations.real
    dim = np.size(rho0, 0)
    rho = cp.Variable((dim,dim),hermitian = True)
    if options.initmethod == 1:
        objective = cp.Minimize(cp.norm(rho0 - rho))
    elif options.initmethod == 2:
        objective = cp.Minimize(-cp.lambda_min(rho))

    constraints = [
        cp.abs(cp.trace(observables[i].T * rho) - expectations[i]) <= options.linearconstrainttolerance
        for i in range(len(observables))
    ]

    prob = cp.Problem(objective, constraints)
    dcp = prob.is_dcp()
    #data = prob.get_problem_data('CVXOPT')
    result = prob.solve(solver='CVXOPT',verbose=True,max_iters = 300,abstol=1e-5)   

    if options.verbose and prob.status == 'Infeasible':
        print("**** Warning: step 1 solver exception, closestDensityMatrix status: %s ****\n", prob.status)

    status = [status, str(prob.status)]
    return rho._value, status
#%%%%%%%%%%%%%%%%
# def   closestDensityMatrix1(rho0:np.array,observables:list,expectations:list,options, status)->list:
#     """
#     Возвращает ближайшую матрицу плотности, решая полуопредленную программу
#     """
#     dim = np.size(rho0,0)
#     initmethod = options.initmethod
#     linearconstrainttolerance = options.linearconstrainttolerance
#     rho = cp.Variable((dim,dim),PSD=True)
#     constraints = []
#     for i in range(len(observables)):
#         constraints.append(cp.abs(cp.trace(cp.conj(observables[i].T)@rho)- expectations[i] ) <= linearconstrainttolerance)  
#     if initmethod == 1:
#         obj = cp.Minimize(cp.norm(rho0-rho))
#         prob = cp.Problem(obj,constraints)
#         prob.solve()
#     elif initmethod == 2:
#         obj = cp.Minimize(-cp.lambda_min(rho))
#         prob = cp.Problem(obj,constraints)
#         prob.solve()

    # return [prob.value,prob.status]


def subproblem(rho, observables, expectations, gradf, options, status):
    n = np.size(rho, 1)
    deltaRho = cp.Variable((n,n), hermitian=True)
    objective = cp.Minimize(
        cp.real(
            cp.trace(gradf @ deltaRho)
        )
    )

    constraints = [
        cp.abs(cp.trace(cp.transpose(observables[i]) @ (rho + deltaRho)) - expectations[i]) <= options.linearconstrainttolerance
        for i in range(len(observables))
    ]

    #rho + deltaRho == hermitian_semidefinite(n)
    constraints.append(rho + deltaRho >> 0)
    prob = cp.Problem(objective, constraints)
    try:
        result = prob.solve(solver='CVXOPT',verbose=True)
    except Exception as e:
        result = prob.solve(solver='SCS',verbose=True)
    try:
        if deltaRho.value ==None:
            result = prob.solve(solver='SCS',verbose=True)
    except Exception as e:
        print('good')
    if options.verbose and prob.status == 'Infeasible':
        print("**** Warning: step 1 solver exception, closestDensityMatrix status: %s ****\n", prob.status)

    status = [status, str(prob.status)]
    return deltaRho.value, status


def step1SolverAsymptotic(rho0, keyMap, observables, expectations, krausOperators, options):

    # array to store cvx status for debugging
    status = []
    default_options = get_default_opt('Asymptotic')

    # Проверяем, все ли параметры заданы
    for option_name in default_options.option_names:
        if not hasattr(options, option_name):
            setattr(options, option_name, getattr(default_options, option_name))
            print(f"**** solver 1 using default {option_name}: {getattr(default_options, option_name)} ****")

    options.verbose = 1 if options.verbose == 'yes' else 0

    fval = 0
    gap = np.inf

    # match options.removeLinearDependence:
    #     case 'rref':
    #         observables, independentCols = removeLinearDependence(observables)
    #         expectations = expectations[independentCols]
    #     case 'qr':
    #         # observables, independentCols = removeLinearDependenceQR(observables)
    #         # expectations = expectations(independentCols)
    #         pass
    if options.removeLinearDependence =='rref':
        observables, independentCols = removeLinearDependence(observables)
        expectations = expectations[independentCols ]
    
    rho, status = closestDensityMatrix(rho0, observables, expectations, options, status)

    if options.verbose == 1:
        print('calculated closest rho\n')
    eigh_vals,eigh_vec = np.linalg.eig(rho)
    min_val = abs(min(eigh_vals))

    if min_val < 0:
        if options.verbose:
            print('**** Error: minimium eigenvalue less than 0. ****')
            print('**** Try increasing the constraint tolerance. ****')
        return

    for _ in range(options.maxiter):
        if options.verbose == 1:
            print('FW iteration:%d', _)
            tstart_FW = time.perf_counter()
 #checked
        gradf = primalDf(rho, keyMap, krausOperators)
        deltaRho, status = subproblem(rho, observables, expectations, gradf, options, status)

        stepSize = minimize_scalar(lambda t: primalf(rho + t * deltaRho, keyMap, krausOperators),
                                   bounds=(options.linesearchminstep, 1),
                                   method='bounded',
                                   options={'xatol': options.linesearchprecision}).x

        gap = np.trace(np.dot((rho + deltaRho) , gradf)) - np.trace(np.dot(rho ,gradf))
        f1 = primalf(rho + stepSize * deltaRho, keyMap, krausOperators)
        f0 = primalf(rho, keyMap, krausOperators)

        if options.verbose == 1:
            t_FW = time.perf_counter() - tstart_FW
            print('FW iteration time:', t_FW)
            print('projection value:',
                  np.trace(np.dot((rho + deltaRho) , gradf)),
                  ' gap:', gap, ' fvalue:', f1,
                  ' fgap:', f1-f0)

        criteria = options.maxgap_criteria
        if (criteria and (np.abs(gap) <= options.maxgap)) or (criteria and ((f1 - f0) >= -options.maxgap)):
            rho = rho + stepSize * deltaRho
            break

        rho = rho + stepSize * deltaRho

        if _ == options.maxiter and options.verbose:
            print('**** Warning: Maximum iteration limit reached. ****')
    eing_val,eign_vec = np.linalg.eig(rho)
    min_eig  = (min(eing_val))
    if min_eig < 0 and options.verbose:
        print('**** Warning: minimium eigenvalue less than 0. ****')

    fval = primalf(rho, keyMap, krausOperators)
    return [rho, fval, gap, status]
