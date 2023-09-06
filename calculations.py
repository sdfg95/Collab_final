import time
import numpy as np
from help_functions import *
from step1SolverAsymptotic import step1SolverAsymptotic
from step2SolverAsymptotic import step2SolverAsymptotic

def get_keyRate(ProtocolDescription,ChannelModel,leakageEC,SolverOptions,fixed,names):
    tstart_iteration = time.time()
    global_settings = SolverOptions.GlobalSetting()
    solver1 = SolverOptions.Solver1()
    solver2 = SolverOptions.Solver2()
    if(global_settings.verboseLevel>=2):
        solver1.verbose = 'yes'
    else:
        solver1.verbose = 'no'
    #%%%%%%%%%%%%%% call step 1 solver %%%%%%%%%%%%%%


    if solver2.name == 'asymptotic':
        k = ProtocolDescription.dimensions
        l = np.prod(k)
        rho0 = np.eye(l)
        [rho,fval,gap,status] = step1SolverAsymptotic(rho0,ProtocolDescription.keyMap,ProtocolDescription.observables,ChannelModel.expectations,ProtocolDescription.krausOp,solver1)

    elif solver2.name == 'asymptotic_inequality':
        rho0 = np.eye(np.prod(ProtocolDescription.dimensions))
        #[rho,fval,gap,solver1Status] = step1SolverAsymptoticInequality(rho0,protocolDescription.keyMap,protocolDescription.observables,protocolDescription.obsMask,channelModel.expectations,channelModel.expMask,protocolDescription.krausOp,solverOptions.solver1);

    elif solver2.name == 'finite':
        dim = ProtocolDescription.dimensions
        mask = ProtocolDescription.obsMask
        observables = ProtocolDescription.observables
        expectations = ChannelModel.expectations

        #additional parameters that can be read from full input parameter list
        try:
            if 'N' in names:
                N = fixed.N
            if 'ptest' in names:
                ptest = fixed.ptest
            if 'eps=' in names:
                eps=  fixed.eps
            m = N * ptest #signals used for testing
            L = len(mask[mask>0])
            mu = muCheckSimplified(eps.PE,L,m) #check if the user has selected post-selection technique for coherent attack. If so, output warning message.
            if ("postselection" in names) and (fixed.postselection == 1) and (global_settings.verboseLevel>=1):
                if 'physDimAB' in names:
                    physDimAB = fixed.physDimAB
                print('**** using post-selection technique for coherent attack ****\n')
                if(np.log10(np.sum(eps))+(physDimAB**2+1)*np.log10(N+1) >= 0):
                    print('**** security cannot be guaranteed with coherent attack, please retry with a smaller N or eps ****\n')
                    print('**** for current N, eps need to be at least smaller than 1e-%d ****\n',((physDimAB**2+1)*np.log10(N+1)))
                    print("security parameter too large")
                else:
                    print('**** note that the security parameter is now %e ****\n',np.sum(eps)*(N+1)**(physDimAB**2+1))
        except Exception as e:
            print(e)

        # %parse observable-expectation pairs using the masks
        # %uncertain observables are labeled as 1
        uncObs = applyMask(observables,mask,1)
        freqs = applyMask(expectations,mask,1)
        certObs = applyMask(observables,mask,0)
        probs = applyMask(expectations,mask,0 )               
        #check that observables are POVMs
        flagTestPOVM = isPOVM(uncObs)
        if(flagTestPOVM!=1):
            print("**** Error: set of observables not POVM! ****\n")
        rho0 = np.eye(np.prod(ProtocolDescription.dimensions))
        #[rho,fval,_,gap,solver1Status]= step1SolverFinite(rho0,ProtocolDescription.keyMap,uncObs,freqs,certObs,probs,mu,ProtocolDescription.krausOp,SolverOptions.solver1)                  
        #check validity of rho and perform perturbation if not valid
        [rho,_]=perturbation_channel(rho0)

#%%%%%%%%%%%%%% call step 2 solver %%%%%%%%%%%%%%
    solver2Status = []

    if (solver2.name=='asymptotic'):
        print('none step2')
        # N = len(ProtocolDescription.observables)        
        # cons = np.zeros(N)       
        # for i in range(N):         
        #     cons[i] = np.abs(np.real(np.trace(np.dot(rho , ProtocolDescription.observables[i])) - ChannelModel.expectations[i]))    #@ - matrix multiplication 
        # solver2.epsilonprime = max(cons)                
        # [val,solver2Status] = step2SolverAsymptotic(rho, ProtocolDescription.observables,ChannelModel.expectations,[],[], ProtocolDescription.keyMap, ProtocolDescription.krausOp, solver2)

    elif solver2.name=='asymptotic_inequality':

        # %parse observable-expectation pairs using the masks  
        # %uncertain observables are labeled as 1      
        obsMask = ProtocolDescription.obsMask  
        LCertObs = len(obsMask[obsMask==0])
        LUncObs = len(obsMask[obsMask==1]) 
        observables = ProtocolDescription.observables  
        expectations = ChannelModel.expectations   
        CertObs = observables[:LCertObs]   
        UncObs = observables[LCertObs:]
        UncObsNeg = np.zeros(len(UncObs))
        CertExp = expectations[:LCertObs]      
        UncExpL = expectations[LCertObs:LCertObs+LUncObs]  
        UncExpU =expectations[LCertObs+LUncObs:]   
        N = len(ChannelModel.expectations) 
        cons = np.zeros(0, N)          
        for i in range(LCertObs):  
            cons[i] = abs(np.real(np.trace(rho @ CertObs[i]) - CertExp[i])) #? 
        for i in range(LUncObs):   
            cons[i+LCertObs] = np.real(UncExpL[i]-np.trace(rho @ UncObs[i])) #?    
            cons[i+LCertObs+LUncObs] = np.real(np.trace(rho @ UncObs[i]) - UncExpU[i]) #?  
        SolverOptions.solver2.epsilonprime = max(cons) 

        for i in range(len(UncObs)):   
            UncObsNeg[i]=-1*UncObs[i]      
        [val,solver2Status] = step2SolverAsymptotic(rho, CertObs,CertExp,[UncObs,UncObsNeg],[UncExpU,-UncExpL], ProtocolDescription.keyMap, ProtocolDescription.krausOp, solver2)


    elif SolverOptions.solver2.name== 'finite':
            cons = np.zeros((0, (len(certObs))))
            for iBasisElm in range(len(certObs)):
                cons[iBasisElm] = abs(np.real(np.trace(rho @ certObs[iBasisElm]) - probs[iBasisElm]))      
            SolverOptions.solver2.epsilonprime = max(cons)
            #2[val,_,solver2Status] = step2SolverFinite(rho,uncObs,freqs,certObs,probs, ProtocolDescription.keyMap, mu, ProtocolDescription.krausOp, SolverOptions.solver2)

    #%%%%%%%%%%%%%% combine privacy amplification and leakage to form key rate %%%%%%%%%%%%%    
    if(solver1.name=='asymptotic_inequality'):
        # %here we only calculate the case of decoy states using asymptotic_inequality solver 
        # %(considering single-photon contribution
        pSignal=ChannelModel.pSignal
        upperBound = pSignal*fval/np.log(2) - leakageEC  
        FWBound = pSignal*(fval-gap)/np.log(2) - leakageEC
        lowerBound = pSignal*val/np.log(2)  - leakageEC

    elif (solver1.name=='finite'):

        # %finite size key rate
        # %considering collective attack
        d = findParameter("alphabet",names)# %the encoding alphabet size
        n = N*(1-ptest)*sum(ChannelModel.pSift)# %received coding signals
        delta = 2*np.log2(d+3)*np.sqrt(np.log2(2/eps.bar)/n)
        correction = (np.log2(2/eps.EC) + 2*np.log2(2/eps.PA))/N
        upperBound = (1-ptest)*(fval/np.log(2)-delta) - correction - (1-ptest)*leakageEC 
        FWBound = (1-ptest)*((fval-gap)/np.log(2)-delta) - correction - (1-ptest)*leakageEC
        lowerBound = (1-ptest)*(val/np.log(2)-delta) - correction - (1-ptest)*leakageEC

    else:
        #default key rate (asymptotic, single-photon source)
        upperBound = fval/np.log(2) - leakageEC
        FWBound = (fval-gap)/np.log(2) - leakageEC 
        #lowerBound = val/np.log(2)  - leakageEC

    upperBound = max([ upperBound])

    FWBound = max([0.0, FWBound])

    #lowerBound = max([0.0, lowerBound])      

    if(global_settings.verboseLevel>0):
        print(f"upperBound:{upperBound}") #, lowerBound: {lowerBound}\n
    
    t_iteration=time.time() - tstart_iteration

    if(global_settings.verboseLevel>0):
        print(f'iteration time: {t_iteration} \n')
    return [upperBound,FWBound] #[upperBound,FWBound, lowerBound]

