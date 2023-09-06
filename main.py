import preset as  preset
from calculations import get_keyRate
import help_functions as hf
import numpy as np
from matplotlib import pyplot as plt

def calculate(protocol_name:str,scan_param_dict:dict,fixed_dict:dict)->list:
    print('Runing')
    parameters = preset.Parameters()
    if protocol_name == 'BB84':
        Fixed = parameters.Fixed(**fixed_dict)
        hf.pmBB84Description(Fixed) 
        protocolDescription = preset.ProtocolDescription()
        hf.pmBB84Channel(Fixed)
        channelModel = preset.ChannelModel()
        leakageEC = hf.generalEC(channelModel,Fixed)
    elif protocol_name == 'DMCVQKD':
        Fixed = parameters.Fixed(**fixed_dict)
        hf.DMCVheterodyneDescription(Fixed)
        protocolDescription = preset.ProtocolDescription()
        hf.channelModel(protocolDescription,Fixed)
        channelModel = preset.ChannelModel()
        leakageEC = hf.generalEC(channelModel,Fixed)

    solverOptions = preset.SolverOptions()
    Scan = parameters.Scan(**scan_param_dict)
    parameters_scan_val =  Scan.value
    scan_name = Scan.name

    dimensions = parameters_scan_val.shape
    N = np.prod(dimensions)# %total dimension of data to scan

    lowerBound = np.zeros(N)
    upperBound = np.zeros(N)
    FWBound = np.zeros(N)

    optimize= parameters.Optimize()
    optimize_name = optimize.name
    optimize_list = optimize.value
    p_lower = optimize_list[0]
    p_start = optimize_list[1]
    p_upper = optimize_list[2]
    isOptimizing = False

    SolverOptions_global = solverOptions.GlobalSetting()
    for i in range(N):
        if(SolverOptions_global.verboseLevel >= 1):
            print('main iteration: {} \n'.format(i))
        if protocol_name == 'DMCVQKD':    
            #%evaluate the protocol description, channel model, and leakage
            if scan_name == 'ETA':
                thisfixed = Fixed
                thisfixed.ETA = parameters_scan_val[i]
                hf.DMCVheterodyneDescription(thisfixed)
                thisProtocolDescription = preset.ProtocolDescription()
                hf.channelModel(thisProtocolDescription,thisfixed)
                thisChannelModel = preset.ChannelModel()

                thisLeakageEC=hf.generalEC(thisChannelModel,thisfixed)

            #names = ["amp_ps","phase_ps","cutoffN","recon","eta","noise","alphaValue","f"]
            elif scan_name == 'AMP_PS':
                thisfixed = Fixed
                thisfixed.AMP_PS = parameters_scan_val[i]
                hf.DMCVheterodyneDescription(thisfixed)
                thisProtocolDescription = preset.ProtocolDescription()

                hf.channelModel(thisProtocolDescription,thisfixed)
                thisChannelModel = preset.ChannelModel()
                
                thisLeakageEC=hf.generalEC(thisChannelModel,thisfixed)

            elif scan_name == 'PHASE_PS':
                thisfixed = Fixed
                thisfixed.PHASE_PS = parameters_scan_val[i]
                hf.DMCVheterodyneDescription(thisfixed)
                thisProtocolDescription = preset.ProtocolDescription()

                hf.channelModel(thisProtocolDescription,thisfixed)
                thisChannelModel = preset.ChannelModel()
                
                thisLeakageEC=hf.generalEC(thisChannelModel,thisfixed)

            elif scan_name == 'CUTOFFN':
                thisfixed = Fixed
                thisfixed.CUTOFFN = parameters_scan_val[i]

                hf.DMCVheterodyneDescription(thisfixed)
                thisProtocolDescription = preset.ProtocolDescription()

                hf.channelModel(thisProtocolDescription,thisfixed)
                thisChannelModel = preset.ChannelModel()
                
                thisLeakageEC=hf.generalEC(thisChannelModel,thisfixed)
            elif scan_name == 'ALPHAVALUE':
                thisfixed = Fixed
                thisfixed.ALPHAVALUE = parameters_scan_val[i]

                hf.DMCVheterodyneDescription(thisfixed)
                thisProtocolDescription = preset.ProtocolDescription()

                hf.channelModel(thisProtocolDescription,thisfixed)
                thisChannelModel = preset.ChannelModel()
                
                thisLeakageEC=hf.generalEC(thisChannelModel,thisfixed)

        elif protocol_name == 'BB84':

            if scan_name == 'ED':
                thisfixed = Fixed
                thisfixed.ED = parameters_scan_val[i]
                hf.pmBB84Description(thisfixed)
                thisProtocolDescription = preset.ProtocolDescription()
                hf.pmBB84Channel(thisfixed)
                thisChannelModel = preset.ChannelModel()                
                thisLeakageEC=hf.generalEC(thisChannelModel,thisfixed)

            elif scan_name == 'PZ':
                thisfixed = Fixed
                thisfixed.PZ = parameters_scan_val[i]
                hf.pmBB84Description(thisfixed)
                thisProtocolDescription = preset.ProtocolDescription()
                hf.pmBB84Channel(thisfixed)
                thisChannelModel = preset.ChannelModel()                
                thisLeakageEC=hf.generalEC(thisChannelModel,thisfixed)

        #%%%%%%%%%%%%%% process single-point parameter %%%%%%%%%%%%%%

       # %%%%%%%%%%%%%% optimize parameter (optional) %%%%%%%%%%%%%%
        
        #%optimization of parameters (if parameters.optimize is non-empty)
        if (isOptimizing):
            #%wrap the key rate function as a single-input function of "p" (numerical array of optimizable parameters)
            rateFunction = getKeyRate_wrapper(parameters.names,hf.reorder(Fixed,parameters.order),protocolDescription,channelModel,leakageEC,solverOptions); #convert getKeyRate as a single-value function f(p) for optimization
            if(solverOptions.globalSetting.verboseLevel >= 1):
                print('begin optimization\n')
            if(solverOptions.optimizer.name == 'bruteForce'):
                p_optimal = hf.bruteForceSearch(rateFunction,p_lower,p_upper,solverOptions.Optimizer())
            elif(solverOptions.optimizer.name == 'coordinateDescent'):
                p_optimal = hf.coordinateDescent(rateFunction,p_start,p_lower,p_upper,solverOptions.Optimizer())
            elif (solverOptions.optimizer.name =='gradientDescent'):
                p_optimal = hf.gradientDescent(rateFunction,p_start,p_lower,p_upper,solverOptions.Optimizer())
            elif (solverOptions.optimizer.name == 'localSearch_Adam'):
                p_optimal = hf.localSearch_Adam(rateFunction,p_start,p_lower,p_upper,solverOptions.Optimizer())

            if(solverOptions.globalSetting.verboseLevel >= 1):
                print('finished optimization\n')

        else:
            p_optimal = []


        #%%%%%%%%%%%%%% evaluate descriptions %%%%%%%%%%%%%%
        
        #%generation of single-row parameter list (a cell array)
        # if(len(p_optimal)==0):
        #     p_full = [p_scan,p_fixed,p_optimal] #concatenate with p_fixed
        #p_full = hf.reorder(p_full,parameters.order)


        

        #%%%%%%%%%%%%%% perform calculation %%%%%%%%%%%%%%
        
        #%calculate key rate by calling the solver module
        #%note that the full parameter list is also passed into the function for the solver to optionally directly access (e.g. security parameter eps, data size N, etc.).
        [upperBound[i],FWBound[i]] = get_keyRate(thisProtocolDescription,thisChannelModel,thisLeakageEC,solverOptions,thisfixed,parameters.names)
    return [upperBound,parameters_scan_val]

#%helper function used for optimization algorithm
def  getKeyRate_wrapper(names,p_full,protocolDescription,channelModel,leakageEC,solverOptions):

    thisProtocolDescription=protocolDescription(names,p_full)
    thisChannelModel=channelModel(thisProtocolDescription,names,p_full)
    thisLeakageEC=leakageEC(thisChannelModel,names,p_full)
    
    solverOptions.solver1.maxiter = 3 #reduce precision for faster speed when optimizing
    solverOptions.globalSetting.verboseLevel = 0
    
    [lowerBound,_,_] = get_keyRate(thisProtocolDescription,thisChannelModel,thisLeakageEC,solverOptions,p_full,names);
    
    return lowerBound


start = 0
stop = 80
step = 10
#result = calculate('BB84',{'name':'ED','start':0,'stop':0.2,'step':0.02},{'PZ':0.4,'F':0.95})
result = calculate('DMCVQKD',{'name':'ETA','start':0,'stop':80,'step':10},{'CUTOFFN':4,'NOISE':0.005})
#построение графика
key, parametr = result
#key[key<0] =0
plt.plot(parametr,key)
plt.show()