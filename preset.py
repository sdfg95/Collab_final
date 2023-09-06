import numpy as np
# 1.parameter to scan over #
#must name at least one parameter to scan (can be a single-point array if only interested in a fixed value)

class Parameters:
    names = ["amp_ps","phase_ps","cutoffN","recon","eta","noise","alphaValue","f"]
    class Scan():
        def __init__(self,name='ETA', start=0,stop=180,step=10) -> None:
            self.name = name
            if name == 'ETA':
                 self.value = 10**(-0.02*np.arange(start,stop,step)) #channel transmittance
            else:
                self.value = np.arange(start,stop,step)


    
    class Fixed(): 
        def __init__(self,AMP_PS =0 ,PHASE_PS =0,CUTOFFN=3,RECON=1,NOISE=0.005,ALPHAVALUE=0.35,ETA=1,F=0.95,PZ=0.5,ED = 0.5) -> None:
            self.AMP_PS = AMP_PS
            self.PHASE_PS = PHASE_PS
            self.CUTOFFN = CUTOFFN
            self.RECON = RECON
            self.NOISE = NOISE
            self.ALPHAVALUE = ALPHAVALUE
            self.ETA = ETA
            self.F = F
            self.PZ = PZ
            self.ED = ED
        # AMP_PS = 0 #amplitude post-selection
        # PHASE_PS = 0 #phase post-selection
        # CUTOFFN = 12 #photon number cutoff
        # RECON = 1 #use reverse reconciliation, 1 for on
        # NOISE = 0.005 #excess noise ksi
        # ALPHAVALUE = 0.35 #signal amplitude
        # ETA = 1
        # F = 0.95 #error-correction efficiency - note that for CVQKD, by convention it is called beta and <1
    
    class Optimize():
        # name = 'AMP_PS'
        # Value = [0.2,0.5,0.7] 
        def __init__(self,name='AMP_PS',value=[0.2,0.5,0.7] ) -> None:
            self.name = name
            self.value = value
        
    #optional; declaring optimizable parameters automatically invokes local search optimizers
    #must be in the format of [lowerBound, initialValue, upperBound]


class SolverOptions:

    class GlobalSetting():
        def __init__(self,name='cvxopt',precision = 'high',verboseLevel = 2) -> None:
            self.cvxSolver = name
            self.cvxPrecision = precision
            self.verboseLevel = 2
        # cvxSolver = 'sdpt3'
        # cvxPrecision = 'high'
        # verboseLevel = 2
        #%output level:
        #0.output nothing (except errors)
        #1.output at each main iteration
        #2.output at each solver 1 FW iteration
        #3.show all SDP solver output
        
    
    class Optimizer():
        def __init__(self,name='coordinateDescent',linearResolution= 3,maxIterations= 10,linearSearchAlgorithm ='iterative', iterativeDepth=2,maxSteps = 10, optimizerVerboseLevel = 1) -> None:
            self.name = name
            self.linearResolution = linearResolution
            self.maxIterations = maxIterations
            self.iterativeDepth = iterativeDepth
            self.linearSearchAlgorithm = linearSearchAlgorithm
            self.maxSteps = maxSteps
            self.optimizerVerboseLevel = optimizerVerboseLevel
        # name = 'coordinateDescent' #choose between 'coordinateDescent' and 'bruteForce'
        # linearResolution = 3 #resolution in each dimension (for brute force search and coordinate descent)
        # maxIterations = 10 #max number of iterations (only for coordinate descent)
        # linearSearchAlgorithm = 'iterative' #%choose between built-in 'fminbnd' and custom 'iterative' algorithms for linear search (only for coordinate descent)
        # iterativeDepth = 2 #choose depth of iteration levels; function count = depth * linearResolution (only for coordinate descent and if using 'iterative')
        # maxSteps = 10 #max number of steps (only for gradient descent and ADAM)
        # optimizerVerboseLevel = 1

    class Solver1():
        def __init__(self,name = 'asymptotic',maxgap = 1e-6,maxiter = 1,initmethod = 2,linearconstrainttolerance = 1e-10,linesearchprecision = 1e-20, linesearchminstep = 1e-3,maxgap_criteria =  True,removeLinearDependence = 'rref') -> None:
            self.name = name
            self.maxgap = maxgap
            self.maxiter = maxiter
            self.initmethod = initmethod
            self.linearconstrainttolerance = linearconstrainttolerance
            self.linesearchminstep = linesearchminstep
            self.linesearchprecision = linesearchprecision
            self.maxgap_criteria = maxgap_criteria
            self.removeLinearDependence = removeLinearDependence
        # name = 'asymptotic'
        # maxgap = 1e-6 #1e-6 for asymptotic, 2.5e-3 for finite
        # maxiter = 1
        # initmethod = 2 #minimizes norm(rho0-rho) or -lambda_min(rho), use method 1 for finite size, 2 for asymptotic v1
        # linearconstrainttolerance = 1e-10
        # linesearchprecision = 1e-20
        # linesearchminstep = 1e-3
        # maxgap_criteria =  True #true for testing gap, false for testing f1-f0
        # removeLinearDependence = 'rref' #'qr' for QR decomposition, 'rref' for row echelon form, empty '' for not removing linear dependence
    
    class Solver2():
        def __init__(self,name='asymptotic', epsilon=0,epsilonprime = 1e-12) -> None:
            self.name = name
            self.epsilon = epsilon
            self.epsilonprime = epsilonprime
        # name = 'asymptotic'
        # epsilon = 0
        # epsilonprime = 1e-12
    
class ProtocolDescription():
    observables = 0
    krausOp = 0
    keyMap = 0
    dimensions = 0
    probList = 0


class ChannelModel():
    expectations = 0
    expectations = 0
    probDist = 0
    pSift = 0
    RECON = 1
    isCVQKD = 1
    errorRate = 0
   