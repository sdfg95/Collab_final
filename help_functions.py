import numpy as np
from pandas import array
from scipy.special import gamma
from scipy.special import gammaincc
from scipy.linalg import sqrtm,logm
from scipy.integrate import dblquad
from scipy import optimize
import math as m
import sympy as sy
import cvxpy as cp
import preset
from math import floor
from preset import  ProtocolDescription , ChannelModel
def zProjector(dim:int,index:int)->np.array:
    """
    Функция предназначена для создания проектора на стостояние, 
    те функция создает матрицу, где ненулевой элеент с индексом(index,index)
    """
    if index>dim:
        index = dim-1
    vec = np.zeros((dim,1))
    vec[index,0] = 1
    return vec*vec.T

def hermitianBasis(dim:int)->np.array:
    """
    Создает ортонормированный базис эрмитовых операторов размерности dim, выходом функции является 
    матрица размерности (dim,dim), а элементами матрицы являются матрицы размерности (dim,dim)
    """
    basis = np.zeros((dim,dim),dtype=list)
    for i in range(dim):
        vec = np.zeros((dim,dim),dtype=complex)
        vec[i,i] = 1 
        basis[i,i] = vec

    for k in range(1,dim):
        for j in range(k):
            vec = np.zeros((dim,dim),dtype=complex)
            vec[k,j] = 1j
            vec[j,k] = -1j
            basis[k,j] = vec/np.sqrt(2)

            vec1 = np.zeros((dim,dim),dtype=complex)
            vec1[k,j] = 1
            vec1[j,k] = 1
            basis[j,k] = vec1/np.sqrt(2)

    return np.reshape(basis,(dim*dim))

def regionOps(N:int,phase_ps:float,amp_ps:float) -> array:
    """
    Функция возвращает операторы области значений 
    """
    regionOp00=np.zeros((N+1,N+1),dtype=complex)
    regionOp01=np.zeros((N+1,N+1),dtype=complex)
    regionOp10=np.zeros((N+1,N+1),dtype=complex)
    regionOp11=np.zeros((N+1,N+1),dtype=complex)

    for l in range(0,N+1):
        for k in range(l+1,N+1):

            regionOp00[l,k] = 2*m.sin((k-l)/4*(m.pi-4*phase_ps))/((k-l)*m.pi*2*m.sqrt(m.factorial(l)*m.factorial(k)))*gammaincc((l+k+2)/2,amp_ps**2)*gamma((l+k+2)/2)
            regionOp00[k,l] =np.conj(regionOp00[l,k])
            regionOp01[l,k] = (1j*(np.exp(-1j/4*(k-l)*(3*m.pi-4*phase_ps))-np.exp(-1j/4*(k-l)*(m.pi+4*phase_ps))))/((k-l)*m.pi*2*m.sqrt(m.factorial(l)*m.factorial(k)))*gammaincc((l+k+2)/2,amp_ps**2)*gamma((l+k+2)/2)
            regionOp01[k,l] = np.conj(regionOp01[l,k])
            regionOp10[l,k] =(1j*(np.exp(1j/4*(k-l)*(3*m.pi+4*phase_ps))-np.exp(1j/4*(k-l)*(5*m.pi-4*phase_ps))))/((k-l)*m.pi*2*m.sqrt(m.factorial(l)*m.factorial(k)))*gammaincc((l+k+2)/2,amp_ps**2)*gamma((l+k+2)/2)
            regionOp10[k,l] = np.conj(regionOp10[l,k])
            regionOp11[l,k] =  (1j*(np.exp(-1j/4*(k-l)*(7*m.pi-4*phase_ps))-np.exp(-1j/4*(k-l)*(5*m.pi+4*phase_ps))))/((k-l)*m.pi*2*m.sqrt(m.factorial(l)*m.factorial(k)))*gammaincc((l+k+2)/2,amp_ps**2)*gamma((l+k+2)/2)
            regionOp11[k,l] = np.conj(regionOp11[l,k])

        regionOp00[l,l] = (1/4-phase_ps/m.pi)*gammaincc(l+1,amp_ps**2)*gamma(l+1)/m.factorial(l)
        regionOp01[l,l] =  (1/4-phase_ps/m.pi)*gammaincc(l+1,amp_ps**2)*gamma(l+1)/m.factorial(l)
        regionOp10[l,l] = (1/4-phase_ps/m.pi)*gammaincc(l+1,amp_ps**2)*gamma(l+1)/m.factorial(l)
        regionOp11[l,l] =  (1/4-phase_ps/m.pi)*gammaincc(l+1,amp_ps**2)*gamma(l+1)/m.factorial(l)
    return [regionOp00,regionOp01,regionOp10,regionOp11]

def zket(dim:int, index:int)->np.array:
    """
    Выводит кет-вектор $\ket{j}$ в вычислительном базисе (z-базисе)
    при индексе = j, считая от 1 до размерности = dim.
    """
    idMatrix = np.eye(dim,dim)
    return idMatrix[:,index].reshape(dim,1)

def binaryEntropy(probability:float)->float:
    """
    Функция для вычисления бинарной энтропии
    """
    if probability == 0 or probability == 1:
        return 0 
    else:
        return (- probability * np.log2(probability) - (1-probability) * np.log2(1 - probability))

def calculateEC(prob_dist:np.array)->list:   	# not checked
    """
    Этот код вычисляет H(X|Y), H(Y|X) и IXY по заданной вероятности
    распределение результатов.
    Dведите нормализованную таблицу распределения вероятностей(prob_dist) для
    расчета.
    """
    [nRows,nColumns] = np.array(prob_dist).shape
    px_dist = np.sum(prob_dist,axis=1)
    py_dist = np.sum(prob_dist,axis=0)
    HXY = 0
    HX = 0
    for iRow in range(nRows):
        for jColumn in range(nColumns):
            if prob_dist[iRow,jColumn]!=0:
               HXY = HXY - prob_dist[iRow,jColumn] * np.log2(prob_dist[iRow,jColumn])
        if px_dist[iRow]!=0:
           HX = HX - px_dist[iRow] * np.log2(px_dist[iRow])
    HY = 0
    for jColumn in range(nColumns):
        if py_dist[jColumn] !=0:
            HY = HY - py_dist[jColumn] * np.log2(py_dist[jColumn]); 
    HY_X = HXY - HX
    HX_Y = HXY - HY
    IXY  = HY - HY_X
    return [HX_Y, HY_X, IXY]

def getRho(basis:np.array,fixedParameters:list, freeVariables:list)->np.array: # not checked
    """
    Функция возвращает матрицу плотности rho, используя фиксированные коэффициенты и набор
    коэффициентов, подлежащих оптимизации в заданном эрмитовом базисе.
    """
    nfixedParameters = len(fixedParameters);
    rho = 0;
    for iFixed in range(nfixedParameters):
        rho = rho + np.dot(fixedParameters[iFixed] , basis[iFixed])
    for jFree in range(len(freeVariables)):
        rho = rho + np.dot(freeVariables[jFree],basis[jFree + nfixedParameters])
    return rho

def gramSchmidt(observables:list, expectations:list, dim:float, tolerance:float)->np.array: # not checked
    """
    Функция возвращает базис  в виде матрицы и список фиксированных параметров.
    """
    if observables:
        basis = [observables[0]/np.sqrt(np.trace(np.dot(observables[0].T , observables[0])).real)]
        counter = 1
        fixedParameters = [expectations[0]/np.sqrt(np.trace(np.dot(observables[0].T , observables[0])).real)]
        for iObs in range(1,len(observables)):
            toAddObservable= observables[iObs]
            toAddExpValue = expectations[iObs]
       
            for jBasis in range(0,counter):
                toAddObservable = toAddObservable - np.trace(np.dot(observables[iObs].T , basis[jBasis])) * basis[jBasis]
                toAddExpValue = toAddExpValue - np.trace(np.dot(observables[iObs].T ,basis[jBasis])) * fixedParameters[jBasis]
            
            if np.sqrt((np.trace(np.dot(toAddObservable.T,toAddObservable)).real))>tolerance:
                basis = [basis, toAddObservable/np.sqrt(np.trace(np.dot(toAddObservable.T , toAddObservable)).real)]
                fixedParameters = [fixedParameters, toAddExpValue/np.sqrt((np.trace(np.dot(toAddObservable.T , toAddObservable))).real)]
                counter = counter + 1

        completeBasis = hermitianBasis(dim)
    
        for iObs in range(len(completeBasis)):
            toAddObservable = completeBasis[iObs]
            for jBasis in range(counter):
                toAddObservable = toAddObservable - np.trace(np.dot(completeBasis[iObs].T ,basis[jBasis])) * basis[jBasis]      
            
            if np.sqrt((np.trace(np.dot(toAddObservable.T*toAddObservable))).real)>tolerance:
                basis = [basis,toAddObservable/np.sqrt((np.trace(np.dot(toAddObservable.T,toAddObservable))).real)]
                counter = counter + 1
        
        if counter != dim**2:
            print('Error in finding basis: try to change the tolerance')  
    else:
        basis = []
        fixedParameters = []
    return [basis,fixedParameters]

def logmsafe(A:np.array)->np.array: #checked
    """
    Функция возвращает матрицу на основе входной матрицы плотности
    """
    [C,V] = np.linalg.eigh(A)
    C = (C.real)
    C[C<1e-13]=0
    logD = np.diag(np.log((C.real)))
    logD[logD ==-np.inf] =0
    logD[logD ==np.inf] =0
    print(logD.shape)
    logA = np.dot(V,np.dot(logD,V.T))
    return logA

def traceG(rho:np.array,krausOperators:list)->float:
    """
    Возвращает след матрицы плотности  после прохождения системы.
    """
   # for the case there is a post-selection map.

    gRho = krausFunc(rho,krausOperators) # calculate G(\rho).
    dim = np.size(gRho,0) # get the dimension of G(\rho).
    eigval,eigvec = np.linalg.eig(gRho)
    eigMin = min(eigval) #check the minimum eigenvalue of this density matrix
    if eigMin <= 0: # if the eigenvalue is less than one, do a perturbation by using a  pinching channel.
       epsilon = (1e-14-eigMin)*dim
       gRho = (1-epsilon)*gRho + epsilon*np.eye(dim)/dim

    fval = np.real(np.trace(gRho)) # calculate the gain
    return fval

def  krausFunc(rho:np.array,krausOperators:list,transpose = False)->np.array:
    """
    Функция возвращает совокупное состояние матрицы плотности и операторов Крауса.
    """
    if (transpose == False ):
        if len(np.shape(krausOperators))==3:
            rhoPrime = np.dot(krausOperators[0],np.dot(rho,krausOperators[0].T))
            for i in range(1,np.shape(krausOperators)[0]):
                rhoPrime +=  np.dot(krausOperators[i],np.dot(rho,krausOperators[i].T))
        else:
            rhoPrime = np.dot(krausOperators,np.dot(rho,krausOperators.T))
    elif transpose==True:
        if len(np.shape(krausOperators))==3:
            rhoPrime = np.dot(krausOperators[0].T,np.dot(rho,krausOperators[0]))
            for i in range(1,np.shape(krausOperators)[0]):
                rhoPrime +=  np.dot(krausOperators[i].T,np.dot(rho,krausOperators[i]))
        else:
            rhoPrime = np.dot(krausOperators.T,np.dot(rho,krausOperators))
    return rhoPrime

def lambda_max( Y:np.array )->float: #mb тип данных complex
    """
    Возвращает максимальное собственное значение входной матрицы Y
    """
    if  np.shape(Y)[0] != np.shape(Y)[1]: #ok
        print( 'Input must be a square matrix.' )
    err = Y - Y.T
    eps =  2.22e-16
    Y   = 0.5 * ( Y + Y.T )
    if np.linalg.norm( err, 'fro' )  > 8 * eps * np.linalg.norm( Y, 'fro' ):
        z = np.inf
    else:
        eig_val, eig_vect = max( np.linalg.eigh(  Y  ) )
        z = max(eig_val)
    return z

def lambda_min( Y:np.array )->float:
    """
    Возвращает минимальное собственное значение входной матрицы Y
    """
    return -lambda_max( -Y )

def perturbation_channel(rho:np.array)->list:
    """
    Возвращает матрицу плотности для возмущенного канала, и значение возмущения
    """
    default_perturbation = 1e-14
    dim = np.size(rho,0)
    eing_val,eign_vec = np.linalg.eig(rho)
    eigMin = (min(eing_val))
    #eigMin = lambda_min(rho)
    epsilon = 0
    rho = (rho+rho.T)/2
    rhoPrime = rho
    if   eigMin<=0:
        if np.real(np.trace(rho))  > 0:
            epsilon = (eigMin  * dim)/(eigMin*dim - np.real(np.trace(rho))) + default_perturbation
            if epsilon < 0 or epsilon > 1/(np.exp(1)*(dim-1)):
                print("**** Error: Perturbation failed. The smallest eigenvalue is less than zero. ****\n")
            else:
                rhoPrime = (1-epsilon) * rho + epsilon * np.real(np.trace(rho))*np.eye(dim)/dim
                rhoPrime = (rhoPrime + rhoPrime.T)/2
    return [rhoPrime,epsilon]

def primalDf(rho:np.array,keyMap:np.array,krausOperators:np.array)->np.array:
    """
    Эта функция вычисляет градиент целевой функции основной задачи.
    """

    if  np.size(krausOperators)==0:

        [rho,_]=perturbation_channel(rho)
        
        zRho = 0
        for j in range(np.size(keyMap)):
            zRho = zRho + np.dot(keyMap[j],np.dot(rho,keyMap[j]))
        
        [zRho,_]= perturbation_channel(zRho)
        
        dfval = np.log(rho)-np.log(zRho)
    else:
        gRho = krausFunc(rho,krausOperators)
        
        [gRho,_] = perturbation_channel(gRho)
        zRho = 0
        for j in range(len(keyMap)):
            zRho = zRho + np.dot(keyMap[j],np.dot(gRho,keyMap[j]))

        [zRho,_] = perturbation_channel(zRho)
        
        dfval = krausFunc(logm(gRho)-logm(zRho),krausOperators,True)
    return dfval

def isPOVM(inputPovm:np.array)->int:
    """
    Функция определяет является ли входящая мера POVM или нет.
    """
    nPovmElm = np.size(inputPovm)
    if nPovmElm == 0:
        result = 0
        return result

    dim = np.size(inputPovm[1],0)
    
    sumElms = 0
    
    for iElm in range(nPovmElm):
        sumElms = sumElms + inputPovm[iElm]

    if np.norm(np.eye(dim)-sumElms,2) < 1e-15:
        result = 1
    else:
        result = 0
    return result

def primalDfep(rho:np.array, keyMap:np.array, krausOperators:list)->list:
    """
    Вычисляет функцию $\nabla f_{\epsilon}(\rho)$, где значение $\epsilon$ тщательно подобрано
    """
    if  len(krausOperators)==0:
        [rho,epsilon1] = perturbation_channel(rho)
    
        zRho = 0
        for jMapElm in range(len(keyMap)):
            zRho = zRho + np.dot(keyMap[jMapElm],np.dot(rho , keyMap[jMapElm]))
        
        [zRho,epsilon2] = perturbation_channel(zRho)
        realEpsilon = max(epsilon1, epsilon2)
        Dfval = logm(rho)-logm(zRho)
    else:
        [rho,epsilon1] = perturbation_channel(rho)
        gRho = krausFunc(rho,krausOperators)
       
        [gRho,epsilon2] = perturbation_channel(gRho)
   
        zRho = 0
        
        for jMapElm in range(len(keyMap)):
            zRho = zRho + np.dot(keyMap[jMapElm],np.dot(gRho , keyMap[jMapElm]))
        
        [zRho,epsilon3] = perturbation_channel(zRho)
        Dfval = krausFunc(logm(gRho)-logm(zRho),krausOperators,True)
        realEpsilon = max([epsilon1,epsilon2,epsilon3])
    return [Dfval,realEpsilon]

def primalf(rho:np.array, keyMap:np.array, krausOperators:list):

    """
    # Этот файл содержит целевую функцию основной задачи.
    # % $f(\rho) := D(\mathcal{G}(\rho)||\mathcal{Z}(\mathcal{G}(\rho)))    
    # Syntax:  fval = primalf(rho,keyMap,krausOperators)     
    # Input:    
    # rho  - density matrix shared by Alice and Bo
    # keyMap - Alice's key map PVM (If Alice's key map POVM is not projective, use Naimark's extension)   
    # krausOperators - The Kraus operators for the post-selection map of
    #  Alice and Bob    
    # Output

    # fval - the objective function value.
    """

    [rho,_]=perturbation_channel(rho)

    if len(krausOperators)==0:
        zRho = 0
        for jMapElement in range(len(keyMap)):
            zRho = zRho + np.dot(keyMap[jMapElement],np.dot(rho,keyMap[jMapElement]))
        
        [zRho,_]=perturbation_channel(zRho)

        fval = np.real(np.trace(np.dot(rho,(logm(rho)-logm(zRho)))))
    else:
        gRho = krausFunc(rho,krausOperators)
        
        [gRho,_]=perturbation_channel(gRho)
        dim  = np.size(gRho,1)
        zRho = 0
        for jMapElement in range(len(keyMap)):
            zRho = zRho + keyMap[jMapElement]@gRho@keyMap[jMapElement]
        


        [zRho,_]=perturbation_channel(zRho)
        

        fval = np.real(np.trace(np.dot(gRho,(logm(gRho)-logm(zRho))))) #+0.36
    return fval

def  primalfep(rho:np.array, keyMap:np.array, krausOperators:list)->list:
    """
    Calculate $f_{\epsilon}(\rho)$ function, where $\epsilon$ value is
    carefully chosen. 
    """
    #     defaultoptions.epsilon = 0 % 0<=epsilon<=1/(e(d'-1)), epsilon is related to perturbed channel
    #     defaultoptions.perturbation = 1e-16; % a small value added to the minimum eigenvalue;
    #     if nargin == 4
    #         if ~isfield(options,'epsilon')
    #             options.epsilon = defaultoptions.epsilon;
    #         end
    #         if ~isfield(options,'perturbation')
    #             options.perturbation = defaultoptions.perturbation;
    #         end
    #     else 
    #         options = defaultoptions;
    #     end
    if len(krausOperators)==0:
       
        dim = np.size(rho,0)
        [rho,epsilon1] = perturbation_channel(rho)

        zRho = 0
        for jMapElm in range(len(keyMap)):
            zRho = zRho + np.dot(keyMap[jMapElm],np.dot(rho , keyMap[jMapElm]))
        
        [zRho,epsilon2] = perturbation_channel(zRho)
        realEpsilon = max(epsilon1, epsilon2);
        fval = np.real(np.trace(np.dot(rho ,(logm(rho) - logm(zRho)))))
    
    else:      
        [rho,epsilon1] = perturbation_channel(rho)
        gRho = krausFunc(rho,krausOperators)
        [gRho,epsilon2] = perturbation_channel(gRho)

        zRho = 0
        
        for jMapElm in range(len(keyMap)):
            zRho = zRho + np.dot(keyMap[jMapElm],np.dot(gRho , keyMap[jMapElm]))
        [zRho,epsilon3] = perturbation_channel(zRho)
        realEpsilon = max([epsilon1,epsilon2,epsilon3])
        fval = np.real(np.trace(np.dot(gRho , (logm(gRho) - logm(zRho)))))
    return [fval,realEpsilon]

def reducedRho1(op:np.array, dim1:int, dim2:int)->np.array:
    """
    Возвращает усеченную матрицу плотности.
    """
    rho1= np.zeros((dim1*dim2,dim1*dim2))
    for i in range(dim2):
        zj = np.kron(np.eye(dim1),zket(dim2,i))
        rho1 = rho1 + np.dot(zj.T , np.dot(op * zj) )
    return rho1

def reducedRho2(op:np.array, dim1:int, dim2:int)->np.array:
    """
    Возвращает усеченную матрицу плотности.
    """
    rho2=np.zeros((dim2*dim1,dim2*dim1))
    for i in range(dim1):
        zj = np.kron(zket(dim1,i),np.eye(dim2)) 
        rho2 = rho2 + np.dot(zj.T ,np.dot(op , zj))
    return rho2

# def  removeLinearDependence(dependentSet:list)->list:
#     """
#     Удаляет линейно зависимые матрицы из набора квадратных матриц.
#     """
#     dim = np.size(dependentSet[0])
#     shape = np.size(dependentSet)
#     reshapedDependentSet = np.zeros((shape,shape))
#     for iEntry in range(np.size(dependentSet)):
#         reshapedDependentSet[iEntry] = np.reshape(dependentSet[iEntry],(dim,dim))
#     reshapedDependentSet = reshapedDependentSet.T

#     [_,independentCols] = sy.Matrix().rref(reshapedDependentSet)
#     independentSet = dependentSet(independentCols)
#     return [independentSet,independentCols]

def removeLinearDependence(dependentSet:list)->list:
    dim1 = np.shape(dependentSet[0])[1]
    dim2 = np.shape(dependentSet[0])[0]
    reshapedDependentSet = []
    for iEntry in range(len(dependentSet)):
        reshapedDependentSet.append (np.reshape(dependentSet[iEntry],(1,dim1*dim2)))
    reshapedDependentSet = np.vstack(reshapedDependentSet).T
    [_,independentCols] = sy.Matrix(reshapedDependentSet).rref()
    dependentSet = np.array(dependentSet)
    independentCols = np.array(independentCols)
    independentSet = dependentSet[independentCols]
    return [independentSet,independentCols]

def   closestDensityMatrix(rho0:np.array,observables:list,expectations:list)->list:
    """
    Возвращает ближайшую матрицу плотности, решая полуопредленную программу
    """
    dim = np.size(rho0,0)
    initmethod = preset.SolverOptions.Solver1.initmethod
    linearconstrainttolerance = preset.SolverOptions.Solver1.linearconstrainttolerance
    rho = cp.Variable((dim,dim),PSD=True)
    constraints = []
    for i in range(len(observables)):
        constraints.append(cp.abs(cp.trace(cp.conj(observables[i].T)@rho) ) <= linearconstrainttolerance)  
    if initmethod == 1:
        obj = cp.Minimize(cp.norm(rho0-rho))
        prob = cp.Problem(obj,constraints)
        prob.solve()
    elif initmethod == 2:
        obj = cp.Minimize(-cp.atoms.lambda_min.lambda_min(rho))
        prob = cp.Problem(obj,constraints)
        prob.solve()

    return [prob.value,prob.status]

def   DMCVheterodyneDescription(Fixed): #checked
    """
    Функция, задающая модель протокла на непрерывных переменных с дискртеной модуляцией, задает операторы
    Крауса, наблюдаемые, распределение вероятностей, размерность Гильбертова пространства.
    """
    # alphaList =[alphaValue,1i*alphaValue,-alphaValue,-1i*alphaValue];
    amp_ps = Fixed.AMP_PS
    phase_ps = Fixed.PHASE_PS
    cutoffN = Fixed.CUTOFFN
    recon = Fixed.RECON
    dimA = 4;
    probList = 1/dimA*np.ones(dimA)
    dimB = cutoffN+1
    
    regionOp = regionOps(cutoffN,phase_ps,amp_ps)
    R00=regionOp[0]
    R01=regionOp[1]
    R10=regionOp[2]
    R11=regionOp[3]
    
    K00 = np.kron(np.eye(dimA*dimB),np.diag([1,0,0,0]))
    K01 = np.kron(np.eye(dimA*dimB),np.diag([0,1,0,0]))
    K10 = np.kron(np.eye(dimA*dimB),np.diag([0,0,1,0]))
    K11 = np.kron(np.eye(dimA*dimB),np.diag([0,0,0,1]))
    keyMap = [K00,K01,K10,K11]
    if (recon==0):# direct reconciliation 
        Rpass = sqrtm(R00+R01+R10+R11)
        A00=np.kron(np.kron(np.diag([1,0,0,0]),Rpass),np.array([[1],[0],[0],[0]]).T)
        A01=np.kron(np.kron(np.diag([0,1,0,0]),Rpass), np.array([0,1,0,0]).T)
        A10=np.kron(np.kron(np.diag([0,0,1,0]),Rpass), np.array([0,0,1,0]).T)
        A11=np.kron(np.kron(np.diag([0,0,0,1]),Rpass), np.array([0,0,0,1]).T)
        krausOp = A00+A01+A10+A11
    elif(recon==1):    # reverse reconciliation 
        B00 = np.kron(np.kron(np.eye(dimA),sqrtm(R00)),np.array([[1],[0],[0],[0]]))
        B01 = np.kron(np.kron(np.eye(dimA),sqrtm(R01)),np.array([[0],[1],[0],[0]]))
        B10 = np.kron(np.kron(np.eye(dimA),sqrtm(R10)),np.array([[0],[0],[1],[0]]))
        B11 = np.kron(np.kron(np.eye(dimA),sqrtm(R11)),np.array([[0],[0],[0],[1]]))
        krausOp = B00+B01+B10+B11
  
    # Constraints

# %     rhoA = zeros(dimA);
# %         
# %     %known rhoA from Alice's choices of signal states and probabilities
# %     %(gram matrix)
# %     
# %     for i=1:dimA
# %         for j=1:dimA
# %             rhoA(i,j)=sqrt(prob(i)*prob(j))*exp(-0.5*(abs(alphaList(i))^2+abs(alphaList(j))^2))*exp(conj(alphaList(j))*alphaList(i));
# %         end
# %     end
    projectorA = np.zeros((dimA),dtype=object)
    for i in range(dimA):
        projectorA[i] = zProjector(dimA,i)/probList[i]

    basis = hermitianBasis(dimA)
    Observables = []
 
    for i in range(len(basis)):
        b = basis[i]
        k = np.kron(b,np.eye(dimB))
        Observables.append(k)

    
    
    #create ladder operators
    line = np.array([ np.sqrt(i) for i in  range(1,cutoffN+1)])
    Raise = np.diag(line,-1)
    lower = np.diag(line,1)
    X = (Raise+lower)/np.sqrt(2)
    P = 1j*(Raise-lower)/np.sqrt(2)
    d = np.matmul(Raise,Raise)+np.matmul(lower,lower)
    N_op = np.diag(range(cutoffN+1))

    #photon_number=(abs(alpha).^2)*eta+noise/2;
    
    for i in range(dimA):
        Observables.append(np.kron(projectorA[i],N_op))
    
    # %X and P constraints
    # %exp_x=sqrt(2)*real(alpha*sqrt(eta)); 
    # %exp_p=sqrt(2)*imag(alpha*sqrt(eta)); 
    # %exp_d=eta*(alpha.^2+conj(alpha).^2);
    
    for i in range(dimA):
        Observables.append(np.kron(projectorA[i],X))
        Observables.append(np.kron(projectorA[i],P))
        Observables.append(np.kron(projectorA[i],d))

    ProtocolDescription.observables = Observables
    ProtocolDescription.krausOp = krausOp
    ProtocolDescription.keyMap = keyMap
    ProtocolDescription.dimensions = [dimA,dimB]
    ProtocolDescription.probList = probList

def channelModel (ProtocolDescription,fixed):
    """
    Функция, задающая модель канала для КРКНП с дискртеной модуляцией.
    """
    # %user should supply the list of parameters usee in this description/channel file
    # %this list varNames should be a subset of the full parameter list declared in the preset file
    # %parameters specified in varNames can be used below like any other MATLAB variables
    # varNames=["eta","noise","alphaValue","recon","phase_ps","amp_ps"];
    
    # %%%%%%%%%%%%%%%%%%%%% interfacing (please do not modify) %%%%%%%%%%%%%%%%%%%%%%%%%

    expectations = []
    expMask = []
    #    %%%%%%%%%%%%%%%%%%%%% user-supplied channel model begin %%%%%%%%%%%%%%%%%%%%%%%%%
        
    dimA = ProtocolDescription.dimensions[0]
    dimB = ProtocolDescription.dimensions[1]
    
    alphaList =np.array([fixed.ALPHAVALUE,1j*fixed.ALPHAVALUE,-fixed.ALPHAVALUE,-1j*fixed.ALPHAVALUE])
    probList = 1/dimA*np.ones(dimA)
    
    #  %% Constraints

    rhoA = np.zeros((dimA,dimA),dtype=complex)
    # %known rhoA from Alice's choices of signal states and probabilities
    # %(gram matrix)
    
    for i in range(dimA):
        for j in range(dimA):
            rhoA[i,j]=np.sqrt(probList[i]*probList[j])*np.exp(-0.5*(abs(alphaList[i])**2+abs(alphaList[j])**2))*np.exp(np.conj(alphaList[j])*alphaList[i])


    projectorA=np.zeros(dimA,dtype=object)
    for i in range(dimA):
        projectorA[i] = zProjector(dimA,i)/probList[i]

    basis = hermitianBasis(dimA)
    
    for i in range(len(basis)):
        bas = basis[i]
        matr = np.matmul(rhoA,bas)
        matr1 = (bas*rhoA)
        tr = np.trace(matr)
        expectations.append(tr)

    
    
    
# #    %create ladder operators
# %     line=sqrt(1:cutoffN);
# %     raise=diag(line,-1);
# %     lower=diag(line,1);
# %     X=(raise+lower)./sqrt(2);
# %     P=1i*(raise-lower)./sqrt(2);
# %     d=raise*raise+lower*lower;
# %     N_op=diag(0:cutoffN);

    photon_number=(abs(alphaList)**2)*fixed.ETA+fixed.ETA*fixed.NOISE/2
    
    for i in range(dimA):
        expectations.append(photon_number[i])

    
#    %X and P constraints
    exp_x= np.sqrt(2)*np.real( alphaList*np.sqrt(fixed.ETA))
    exp_p=np.sqrt(2)*np.imag( alphaList*np.sqrt(fixed.ETA))
    exp_d=fixed.ETA*(alphaList**2+np.conj( alphaList)**2)
    
    for i in range(dimA):
        expectations.append(exp_x[i])
        expectations.append(exp_p[i])
        expectations.append(exp_d[i])

    
    beta = np.sqrt(fixed.ETA)* alphaList
    prob_dist = np.zeros((4,4))
    theta_lower = [-np.pi/4+fixed.PHASE_PS,np.pi/4+fixed.PHASE_PS,3*np.pi/4+fixed.PHASE_PS,5*np.pi/4+fixed.PHASE_PS]
    theta_upper = [np.pi/4-fixed.PHASE_PS,3*np.pi/4-fixed.PHASE_PS,5*np.pi/4-fixed.PHASE_PS,7*np.pi/4-fixed.PHASE_PS]
    
    for i in range(4):
        for j in range(4):
            prob_dist[i,j]= dblquad(gaussian,fixed.AMP_PS,np.inf,theta_lower[j],theta_upper[j], args=[i,fixed.ALPHAVALUE,fixed.ETA,fixed.NOISE] ,epsabs = 1e-12)[0]/4

    p_pass = np.sum(sum(prob_dist))
    prob_dist=prob_dist/p_pass
    
 #   %%%%%%%%%%%%%%%%%%%%% user-supplied channel model end %%%%%%%%%%%%%%%%%%%%%%%%%

    ChannelModel.expectations =np.array(expectations) 
    ChannelModel.probDist = prob_dist
    ChannelModel.pSift = [p_pass]
    ChannelModel.recon = fixed.RECON
    ChannelModel.isCVQKD = 1
   

def pmBB84Description(Fixed):
    pz = Fixed.PZ
    dimA = 4
    dimB = 2
    observ = []
    ketPlus = 1/np.sqrt(2)*np.array([[1],[1]])
    ketMinus = 1/np.sqrt(2)*np.array([[1],[-1]])
    krausOpZ1 = zket(2,1)
    krausOpZ = np.kron(np.kron(np.kron(zket(2,0),np.diagflat([1,0,0,0]))+ np.kron(zket(2,1),np.diagflat([0,1,0,0])), np.sqrt(pz)*np.eye(dimB)), [[1],[0]])# % for Z basis
    krausOpX = np.kron(np.kron(np.kron(zket(2,0),np.diagflat([0,0,1,0]))+ np.kron(zket(2,1),np.diagflat([0,0,0,1])),np.sqrt(1-pz) * np.eye(dimB)),[[0],[1]]) # % for X basis
    krausOp = [krausOpZ, krausOpX] 
    #% components for the pinching Z map
    keyProj1 = np.kron(np.diagflat([1,0]), np.eye(dimA*dimB*2))
    keyProj2 = np.kron(np.diagflat([0,1]), np.eye(dimA*dimB*2))
    keyMap = [keyProj1, keyProj2]
    #% Constraints
    basis = hermitianBasis(dimA)
    for iBasisElm in range (len(basis)):
        observ.append(np.kron(basis[iBasisElm], np.eye(dimB)))
    #% Z and X constraints
    observ.append(np.kron(np.diagflat([1,0,0,0]),np.diagflat([0,1])) + np.kron(np.diagflat([0,1,0,0]), np.diagflat([1,0])))
    observ.append(np.kron(np.diagflat([0,0,1,0]),ketMinus * np.transpose(ketMinus)) + np.kron(np.diagflat([0,0,0,1]), ketPlus * np.transpose(ketPlus)))#    %     % Cross terms%     addObservables(kron(diag([1,-1,0,0]), ketPlus * ketPlus' - ketMinus * ketMinus'));%     addObservables(kron(diag([0,0,1,-1]), diag([1,-1])));
    #% Normalization
    observ.append(np.eye(dimA*dimB))
    #%%%%%%%%%%%%%%%%%%%%% user-supplied description end %%%%%%%%%%%%%%%%%%%%%%%%%    
    ProtocolDescription.observables = np.array(observ)
    ProtocolDescription.krausOp = np.array(krausOp)
    ProtocolDescription.keyMap = np.array(keyMap)
    ProtocolDescription.dimensions = [dimA,dimB]

def pmBB84Channel(Fixed):

    # %user should supply the list of parameters used in this description/channel file
    # %this list varNames should be a subset of the full parameter list declared in the preset file
    # %parameters specified in varNames can be used below like any other MATLAB variables
    # varNames=["ed","pz"];
    
    #%%%%%%%%%%%%%%%%%%%%% interfacing (please do not modify) %%%%%%%%%%%%%%%%%%%%%%%%%
    
    expectations = []
    expMask = []
    ed = Fixed.ED
    pz = Fixed.PZ
    #%%%%%%%%%%%%%%%%%%%%% user-supplied channel model begin %%%%%%%%%%%%%%%%%%%%%%%%%
    
    dimA = ProtocolDescription.dimensions[0]
    dimB = ProtocolDescription.dimensions[1]
    
    ketPlus = 1/np.sqrt(2)*np.array([[1],[1]])
    ketMinus = 1/np.sqrt(2)*np.array([[1],[-1]])
    signalStates = np.array([[[1],[0]], [[0],[1]], ketPlus, ketMinus])
    probList = np.array([[pz/2], [pz/2], [(1-pz)/2], [(1-pz)/2]])
    
    # % rho_A constraints
    rhoA = np.zeros((dimA,dimA))
    for jRow  in range(dimA):
        for kColumn  in range(dimA):
            rhoA[jRow,kColumn] = np.sqrt(probList[jRow] * probList[kColumn]) * np.dot(np.transpose(signalStates[kColumn]) , signalStates[jRow])
    rhoA[abs(rhoA)<=1e-15] = 0
    basis = hermitianBasis(dimA)
    for iBasisElm in range(len(basis)):
        expectations.append(np.trace(np.dot(rhoA , basis[iBasisElm])))


#    % Z and X constraints
    expectations.append(pz*ed)
    expectations.append((1-pz)*ed)
    
#     % Cross terms
# %     addExpectations(0);
# %     addExpectations(0);

#     % Normalization
    expectations.append(1)
    
    #%%%%%%%%%%%%%%%%%%%%% user-supplied channel model end %%%%%%%%%%%%%%%%%%%%%%%%%
    
    ChannelModel.expectations = np.array(expectations)
    ChannelModel.errorRate = [ed,ed]
    ChannelModel.pSift = [pz**2,(1-pz)**2]
    ChannelModel.isCVQKD = 0

def gaussian(y,x,i,ALPHAVALUE,ETA,NOISE)->np.array:
    """
    Функция, задающая 
    """
    alphaList =np.array([ALPHAVALUE,1j*ALPHAVALUE,-ALPHAVALUE,-1j*ALPHAVALUE])
    beta = np.sqrt(ETA)* alphaList
    g = ((np.exp(-abs(x*np.exp(1j*y)-beta[i])**2/(1+ETA*NOISE/2))/(np.pi*(1+ETA*NOISE/2))*x))
    return g


def applyMask(a:list,msk:list,label)->list:
    """
    Функция для выбора части ячейки/числового массива, используя другой массив масок той же длины
    # можно использовать для разбора входящих наблюдаемых (ожиданий) на группы с obsMask (expMask)
    """
    if(len(a)!=len(msk)):
        print('**** mask length mismatch! ****\n')
    a_m=[]
    for i in range(len(msk)):
        if(msk[i]==label):
            a_m.append(a[i])
    return a_m


def findParameter(varName,names:list,p:list):
    """
    Gолучить один параметр с заданной меткой "varName" из списка входных параметров p этой итерации
    """
    varValues = findVariables(varName,names,p) #returns a cell array
    
    if (len(varValues)==0):
       print('**** parameter %s not found! ****\n',varName)
       varValue = []
    else:
       varValue = varValues[0]# %retrieve the parameter value (can be a numerical value or an array/matrix)
    return varValue


def hasParameter(varName,names:list):
    """
    Проверяет, существует ли один параметр с заданной меткой "varName" в списке входных параметров p этой итерации.
    """
    found = False

    for j in range(len(names)):
        if ((varName==names[j])):
            found = True
            break #repeated entries will be ignored
    return found

def muCheckSimplified(eps_PE, POVMoutcomes, m):
    mu = np.sqrt(2)*np.sqrt((np.log(1/eps_PE) + POVMoutcomes * np.log(m+1))/m)
    return mu

def findVariables(varNames,names,values):
# %helper function that looks for variables based on varNames,
# %searching from provided list of names and values
# function varValues = findVariables(varNames,names,values)
    
    varValues = []
    counter = 0
    for i in range(len(varNames)):
        for j in range(len(names)):
            if((varNames[i]==names[j])):
                varValues.append(values[j])
                counter = counter + 1
                break #repeated entries will be ignored
    
    if(counter!=len(varNames)):
        print('description/channel parameters not a subset of input parameter list!')
        varValues = [] #error output
    return varValues

def generalEC(ChannelModel,fixed):
    """
    Эта функция возвращает дескриптор функции утечки исправления ошибок для общей формулировки протоколов.
    Возвращаемая функция зависит только от channelModel, который содержит QBER или распределение вероятностей 
    (в зависимости от мелкозернистой/крупнозернистой модели) и вероятность просеивания.

    """
    # %this list varNames should be a subset of the full parameter list declared in the preset file
    
    # %the functions findVariables and addVariables automatically search the input (names,p) for
    # %the parameter values based on varNames, and convert them to MATLAB variables.
    # %from here on the parameters specified in varNames can be used like any other MATLAB variables
    #varValues = findVariables(varNames,names,p)
    #addVariables(varNames,varValues)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #%pSift can be an array (e.g. [pz^2, (1-pz)^2])
    leakageEC = 0
    siftingFactor = (ChannelModel.pSift)

    dimension = len(siftingFactor)
    if(ChannelModel.isCVQKD!=1):
        #%DVQKD
        if(len(ChannelModel.errorRate)!=0):
            #%simple QBER model
            errorRate = ChannelModel.errorRate
            for i in range(dimension):
                leakageEC = leakageEC +  fixed.F * siftingFactor[i] * binaryEntropy(min(0.5,errorRate[i])) 
        else:
            #%model based on probability distribution
            probDist = ChannelModel.probDist
            for i in range(dimension):
                if(dimension == 1):
                    [HX_Y, HY_X, IXY] = calculateEC(probDist)
                else:
                    #%multiple distributions
                    [HX_Y, HY_X, IXY] = calculateEC(probDist[i])
                leakageEC = leakageEC + fixed.F * siftingFactor[i] * HX_Y   
    else:
        # %CVQKD
        # % DR: EC= (IXY+HX_Y)*pSift - f * IXY*pSift, where f<=1.  
        # % RR: EC= (IXY+HY_X)*pSift - f * IXY*pSift, where f<=1. 
        if((ChannelModel.probDist.all()==0)):
            print('need a probability distribution for error correction!')
        probDist = ChannelModel.probDist
        if ChannelModel.RECON == 1:
            #%use reverse reconciliation
            for i in range(dimension):
                if(dimension == 1):
                    [HX_Y, HY_X, IXY] = calculateEC(probDist);
                else:
                    [HX_Y, HY_X, IXY] = calculateEC(probDist[i]);
                leakageEC = leakageEC +(IXY+HY_X)*siftingFactor[i] - fixed.F*IXY*siftingFactor[i]
        else:
            #%use direct reconciliation
            for i in range(dimension):
                if(dimension == 1):
                    [HX_Y, HY_X, IXY] = calculateEC(probDist)
                else:
                    [HX_Y, HY_X, IXY] = calculateEC(probDist[i])
                leakageEC = leakageEC +(IXY+HX_Y)*siftingFactor[i] - fixed.F*IXY*siftingFactor[i]
    return  leakageEC

def coordinateDescent(f,p_start,p_lower,p_upper,options):
    """
    Функция, реализующая координатный спуск, для потимизации параметров системы.
    """
    #implement the algorithm in W Wang, F Xu, and HK Lo. Physical Review X 9 (2019): 041012.

    #process other input options
    if (options.optimizerVerboseLevel==0):
        optimizerVerboseLevel=options.optimizerVerboseLevel
    else:
        optimizerVerboseLevel=1

    if(optimizerVerboseLevel==2):
        linearverbose='iter'
    else:
        linearverbose='off'

    maxIterations=options.maxIterations

    if(options.linearSearchAlgorith==0):
        linearSearchAlgorithm=options.linearSearchAlgorithm
    else:
        linearSearchAlgorithm='fminbnd'

    if(options.iterativeDepth==0):
        depth = options.iterativeDepth
    else:
        depth = 2

    if(options.linearResolution == 0):
        maxFunEvals=options.linearResolution
    else:
        maxFunEvals=6

    if(options.iterationtolerance==0):
        iterationtolerance=options.iterationtolerance
    else:
        iterationtolerance=1e-6
    
    #%process search dimensions (number of variables)
    num_variables=len(p_upper)
    if(len(p_upper)!=len(p_lower) or len(p_start)!=len(p_lower)):
        print('search range dimensions mismatch!\n')
        return
    
    if(optimizerVerboseLevel==1):
        drawWaitbar()
    
    p_optimal=p_start
    v_optimal=f[p_start]
    
    #%coordinate descent algorithm
    for i in range(maxIterations):
        if(optimizerVerboseLevel==2):
            print('coordinate descent iteration: %d\n',i)
        p_local=p_optimal
        v_local=v_optimal
        #%in each iteration, a linear search is performed for each variable in sequence
        for j in range(num_variables):
            if(optimizerVerboseLevel==2):
                print('variable: %d\n',j)
            f_1D = lambda x:x-f[(p_local,x,j)] #@(x) -f[replace_array_at(p_local,x,j)] #%create a 1D "binded" function, note the sign flipping since we're maximizing f by default
            #%use a linear search algorithm
            if((linearSearchAlgorithm=='fminbnd')):
                if(optimizerVerboseLevel==1):
                    currentProgress = ((i-1)*num_variables+(j-1))/(num_variables*maxIterations)
                    barLength = 1/(num_variables*maxIterations)
                    f_update = lambda x,v,state: fminbnd_update(x,v,state,maxFunEvals,currentProgress,barLength)
                    [x,v]=optimize.fminbound(f_1D,p_lower(j),p_upper(j))
                else:
                    [x,v]=optimize.fminbound(f_1D,p_lower(j),p_upper(j))

            else:
                if(optimizerVerboseLevel==1):
                    currentProgress = ((i-1)*num_variables+(j-1))/(num_variables*maxIterations)
                    barLength = 1/(num_variables*maxIterations)
                    f_update = lambda x,v,state: fminbnd_update(x,v,state,maxFunEvals*depth,currentProgress,barLength)
                    [x,v]=fminbnd_iterative(f_1D,p_lower(j),p_upper(j),maxFunEvals,depth,linearverbose,f_update)
                else:
                    [x,v]=fminbnd_iterative(f_1D,p_lower(j),p_upper(j),maxFunEvals,depth,linearverbose);

            v=-v #%maximizing f
            if(v>v_local):
                p_local[j]=x
                v_local=v

            if(optimizerVerboseLevel==2):
                print('[ ')
                for k in range(num_variables):
                    print('%f ',p_local(k))
                print(']   ')
                print('f=%f\n',v_local)
 
        if(abs(v_local-v_optimal)<iterationtolerance):
            if(optimizerVerboseLevel==1):
                clearWaitbar()

            if(optimizerVerboseLevel!=-1):
                print('found local minimum\n')
                v_optimal=v_local

                print('[ ')
                for k in range(num_variables):
                    print('%f ',p_local[k])

                print(']   ')
                print('f=%f\n',v_local)

            break

        p_optimal=p_local
        v_optimal=v_local
        
        
        if(i==maxIterations):
            if(optimizerVerboseLevel==1):
                clearWaitbar()
            
            if(optimizerVerboseLevel!=-1):
                print('reached max iterations\n') 

                print('[ ')
                for k in range(num_variables):
                    print('%f ',p_local[k]) 

                print(']   ')
                print('f=%f\n',v_local)
    return [p_optimal,v_optimal]


# def getOrder(parameters):
#     """
#     Генерирует порядок полей параметров.
#     ожидается, что параметры структуры будут иметь поля "names"
#     вызовите это во время основной итерации для сортировки параметров
#     """
#     names = parameters.names
#     newNames = names
#     order = np.zeros(1,len(names))
    
#     pos = 1
#     fieldlist = fieldnames(parameters)
#     for i in range(len(fieldlist)):
#         field = parameters.fieldlist[i] #%check each of scan/fixed/optimize structs
#         if (isstruct(field)):
#             #%get name of variables
#             varlist = fieldnames(field)
#             for j in len(varlist):
#                 #%one variable
#                 varname = varlist[j]
#                 found = 0
#                 for k in range(len(names)):
#                    if (varname==names[k]):
#                       order[k]=pos
#                       found = 1

#                 if (found==0):
#                    newNames = [newNames,varname]
#                    order = [order,pos]
#                 pos = pos+1
#     return [newNames,order]


def flattenIndices(indices:np.array,dimensions:list)->float:
    """
    вспомогательная функция: возвращает индекс в сглаженном одномерном представлении матрицы в виде массива
    """
    indices = indices - 1 #% convert to 0-based index
    dim = len(dimensions)
    i = 0
    multiple = 1
    for j in range(0,dim,-1):
        i = i + multiple * indices[j]
        multiple = multiple * dimensions[j]

    i = i + 1 #% convert to 1-based index
    return i


def expandIndices(i:int, dimensions:list)->list:
    """
    вспомогательная функция: возвращает индексы в расширенном многомерном матричном представлении массива
    """
    i = i - 1 #% convert to 0-based index
    dim = len(dimensions)
    indices=([])
    residue = i
    for j in range(dim):
        multiple = 1
        for k in range(j+1,dim):
            multiple = multiple * dimensions[k]

        index = floor(residue/multiple)
        residue = (residue*multiple)
        indices.append(index)
    return indices


def  selectCellRow(indices:list,array:list)->list:
    """
    вспомогательная функция: выбирает данную одномерную строку массива ячеек в зависимости от входных индексов
    """
    L = len(array)
    selected = []
    for k in range(L):
       element = array[k]
       selected = [selected,element[indices[k]]];
    return selected


def  getCellDimensions(array:list)->list:
    """
    вспомогательная функция: возвращает размеры (в массиве) массива ячеек
    """
    L = len(array)
    dimensions = []
    if L!=0:
        for k in range(L):
           element = array[k]
           dimensions = [dimensions,len(element)]
    else:
        dimensions=[0]
    return dimensions

def reorder(array:list,order:list)->list:
    #here array is assumed to be a cell array
    ordered=[]
    for i in range(len(array)):
        index=order[i]
        ordered=[ordered,array(index)]
    return ordered

def drawWaitbar():
    """
    Рисует полосы загрузки
    """
    L=50
    print("[")
    for i in range(L):
        print("-")
    print("]")

def clearWaitbar():
    """
    Стирает полосы загрузки
    """
    L=50
    #%clear previous row
    for i in range(L+2):
        print("\b")
def fminbnd_update(x,v,state,totalCount,currentProgress,barLength):
    if state == 'iter':
        print(v.funccount)
        updateWaitbar((v.funccount/totalCount)*barLength+currentProgress)
    stop = False
    return stop

def updateWaitbar(progress):
    L=50
    barlength = (progress*L)
    #clear previous row
    for i in range(L+2):
        print("\b")
    print("[")
    for i in range(barlength):
        print("*")
    for i  in range(L-barlength):
        print("-")
    print("]")
