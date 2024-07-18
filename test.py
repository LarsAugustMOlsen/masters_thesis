import numpy as np
import scipy.io
import scipy.ndimage
from scipy.optimize import minimize
from scipy.special import gammaln, softmax
from scipy.stats import circmean, circvar
import time
import concurrent.futures
import pickle
import gzip
import warnings

def CircleDist(angles, D):
    dx = abs(angles - D)
    dx = np.minimum(dx, 2 * np.pi - dx)
    return dx

def SphericalDist(angles, D):
    D = np.array(D)
    theta1, phi1 = angles[0], angles[1]
    distances = np.zeros_like(D[:, 0])
    for i in range(len(D)):
        theta2, phi2 = D[i, 0], D[i, 1]
        cos_angle = (np.sin(phi1) * np.sin(phi2) +
                     np.cos(phi1) * np.cos(phi2) * np.cos(theta1 - theta2))
        distances[i] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return distances

def getLambdaTwoCircles(a, D):
    tc_preftheta = a[0] % (2.*np.pi)
    tc_prefphi = a[1] % (2.*np.pi)
    tc_beta = a[2]
    tc_h = a[3]
    tc_choose = a[4]
    tc_sigma = np.pi/2
    if len(a) > 5:
        tc_sigma = a[5]
    

    dist1 = CircleDist(tc_preftheta, D[:, 0])
    dist2 = CircleDist(tc_prefphi, D[:, 1])
    bump1 = tc_beta * np.exp(-dist1**2 / (2*(tc_sigma)**2))
    bump2 = tc_beta * np.exp(-dist2**2 / (2*(tc_sigma)**2))
    bump2d = tc_h + bump1 * tc_choose + bump2 * (1 - tc_choose)
    return bump2d      


def logLtc(a, Sj, D):
    H = getLambdaTwoCircles(a, D)
    if H.shape != Sj.shape:
        print(f"Shape mismatch: H shape {H.shape}, Sj shape {Sj.shape}")
    return -np.sum(Sj * H - np.exp(H) - gammaln(Sj+1))

def penaltyForWeights(x):
    penalty = 1 - (x - 0.5)**2
    return penalty

def getscorepp(Sj, estDir):
    global updatingPenaltyWeight

    def objFunc(a):
        return logLtc(a, Sj, estDir) + updatingPenaltyWeight * penaltyForWeights(a[4])

    def objectiveLogging(a):
        obj1 = logLtc(a, Sj, estDir)
        obj2 = penaltyForWeights(a[4])
        return obj1, obj2

    # Initial bounds and parameters
    bds = ((0, 2*np.pi+0.001), (0, 2*np.pi+0.001), (0, None), (None, None), (0, 1))
    a = np.zeros(5)
    fun = 10**10

    # Phase 1: Optimize for all different values
    for theta in np.arange(0, 2*np.pi, np.pi/5.):
        for phi in np.arange(0, 2*np.pi, np.pi/5.):
            result = minimize(objFunc, [theta, phi, 3, -2, 0.49], method='L-BFGS-B', bounds=bds, options={"maxiter":10})
            if (result.fun < fun):
                fun = result.fun
                a = result.x

    # Add the variance
    bds = ((None, None), (None, None), (0, None), (None, None), (0, 1), (None, 10))
    result = minimize(objFunc, [a[0], a[1], a[2], a[3], a[4], np.pi/2], method='L-BFGS-B', bounds=bds,options={"maxiter":10})

    final_obj1, final_obj2 = objectiveLogging(result.x)
    return final_obj1, final_obj2, result

def meanandstd_one_circle(a):
    mean_theta = circmean(a)
    var_theta = circvar(a)
    #print(mean_theta, np.sqrt(var_theta))
    return mean_theta, np.sqrt(var_theta)




def getpreddir(params, S, P=80, NX=3):
    T = S.shape[1]  # Number of time steps
    
    def getlikelihood(Dtry):
        logl = np.zeros(T)
        for i in range(S.shape[0]):  # Iterate over trials
            H = getLambdaTwoCircles(params[i, :], Dtry)
            logl += (S[i, :] * H - np.exp(H) - gammaln(S[i, :] + 1))
        return np.exp(logl)

    particles = np.random.uniform(0, 20 * np.pi, size=(P, T, 2)) % (2. * np.pi)
    Dc = np.zeros((T, 2))  # Update shape to (2, T)
    Dcstd = np.zeros((T, 2))  # Update shape to (2, T) # Update shape to (T, 2) to store std for each dimension

    for xxx in range(NX):
        particles = particles + np.random.normal(0, np.pi / 4, size=(P, T, 2))
        particles %= (2 * np.pi)
        likelihoods = np.zeros((P, T))
        for i in range(P):
            Dt = particles[i, :, :]
            likelihoods[i, :] = getlikelihood(Dt)

        normfact = np.sum(likelihoods, axis=0)
        weights = likelihoods / normfact[np.newaxis, :]

        for t in range(T):
            inds = np.random.choice(np.arange(P), size=P, p=weights[:, t])
            particles[:, t, :] = particles[inds, t, :]
            if xxx == NX - 1:
                angs = particles[:, t, :] + 0.
                mean1, std1 = meanandstd_one_circle(angs[:,0])
                mean2, std2 = meanandstd_one_circle(angs[:,1])
                #print("mean1= ", mean1, "  mean2 = ",mean2)                        
                Dc[t,0],Dc[t,1] = mean1 , mean2
                Dcstd[t,0], Dcstd[t,1] = std1 , std2

                # Check for NaNs in the result
                if np.any(np.isnan(mean1)) or np.any(np.isnan(mean2)) or np.any(np.isnan(std1)) or np.any(np.isnan(std2)):
                            print(f"NaN detected at timestep {t} during iteration {xxx}")
                            #print(f"Angles: {angs}")
                            print(f"Mean and STD result: {mean1, mean2}, {std1,std2}")
    return Dc, Dcstd

def compute_loglj(params, estDir, S, i, j):
    H = getLambdaTwoCircles(params[j, :], estDir)
    loglj = np.sum(S[j, :] * H - np.exp(H) - gammaln(S[j, :] + 1))
    return i, j, loglj


# Load the data
with gzip.open('twocirclesseparable.pkl.gz', 'rb') as f:
    data = pickle.load(f)

hd_sim = data['hd_train']
S = np.array(data['S_train'])
N = len(S[:,0])
T = len(S[0,:])
# Define the global penalty weight
updatingPenaltyWeight = 10000
print(hd_sim.shape)
print(S.shape)


# for i in range(5):
#     final_obj1, final_obj2, result = getscorepp(S[i, :], hd_sim.T)
#     obj1.append(final_obj1)
#     obj2.append(final_obj2)
#     results.append(result)
# params = np.array([res.x for res in results])
def mainFunction(i):
    print("Start nr. ", i, "\n \n")
    np.random.seed(i)
    NumberOfIterations = 30 # How many steps of estimation and maximization is run?
    llvals = np.zeros(NumberOfIterations)
    #preddir = (hd_sim.T + np.random.normal(0,2*np.pi, size = (T,2)))%(2*np.pi) #Define startpoint
    preddir = ((np.random.uniform(size=(T,2)))*1000)%(2*np.pi)
    for i in range(NumberOfIterations):
        print("\n Iteration ",i)
        temp_best_llvals = float('-inf')
        #Estimation step
        results_list = []
        obj1total = 0
        obj2total = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
            results = list(executor.map(getscorepp, S, [preddir] * N))
        for final_obj1, final_obj2, result in results:
            obj1total += final_obj1
            obj2total += final_obj2
            results_list.append(result)
        params = np.array([res.x for res in results_list])
        params2 = params
        allocation = params[:,4]
        weights = params[:,2]
        print("allocations: ",allocation)
        print("weights: ", weights)
    

        #Maximization step
        preddir, predSTD = getpreddir(params2,S)

        with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
            futures = []
            for j in range(N):
                futures.append(executor.submit(compute_loglj, params, preddir, S, i, j))

        for future in concurrent.futures.as_completed(futures):
            i, j, loglj = future.result()
            llvals[i] += loglj
        print("this iterations llvals: " , llvals[i])
        if(llvals[i]>temp_best_llvals):
            temp_best_llvals = llvals[i]
            best_obj1total, best_obj2total, best_params, best_results_list, best_preddir, best_predSTD, best_llvals = obj1total, obj2total, params, results_list, preddir, predSTD, llvals
    print("The best llvals value for this restart is ",best_llvals)
    return i, best_obj1total, best_obj2total, best_params, best_results_list, best_preddir, best_predSTD, best_llvals
best_llvals_last_entry = float('-inf')
best_results = None

for i in range(20):
    i_temp, obj1total_temp, obj2total_temp, results_list_temp, params_temp, preddir_temp, predSTD_temp, llvals_temp = mainFunction(i)
    
    # Check if the current llvals_temp's last entry is the highest
    if llvals_temp[-1] > best_llvals_last_entry:
        best_llvals_last_entry = llvals_temp[-1]
        i, obj1total, obj2total, results_list, params, preddir, predSTD, llvals = i_temp, obj1total_temp, obj2total_temp, results_list_temp, params_temp, preddir_temp, predSTD_temp, llvals_temp



        


print("Final Objective 1:", obj1total)
print("Final Objective 2:", obj2total)
print("Optimization Result:", results_list[-1])

data = {"result":results_list, "final_obj1":obj1total, "final_obj2":obj2total, "params":params,"results_list":results_list, "preddir":preddir, "predSTD":predSTD, "llvals":llvals}
with gzip.open('best_estDir.pkl.gz', 'wb') as f:
    pickle.dump(data, f)