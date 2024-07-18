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



output_file_path = 'output_parallel.txt'
with open(output_file_path, 'w') as output_file:
    # Global variable:
    
    # Sigmoid Function:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Distance between two points on torus
    def ToroidalDist(angles, D):
        D = np.array(D)
        theta = angles[0]
        phi = angles[1]
        dx = abs(D[:, 0] - theta)
        dy = abs(D[:, 1] - phi)
        dx = np.minimum(dx, 2 * np.pi - dx)
        dy = np.minimum(dy, 2 * np.pi - dy)
        distances = np.sqrt(dx**2 + dy**2)
        return distances
    
    # Distance between two points living on a circle
    def CircleDist(angles, D):
        dx = abs(angles - D)
        dx = np.minimum(dx, 2 * np.pi - dx)
        return dx

    # Distance between two points on a sphere
    def SphericalDist(angles, D):
        D = np.array(D)
        theta1, phi1 = angles[0], angles[1]
        distances = np.zeros_like(D[:, 0])
        for i in range(len(D)):
            theta2, phi2 = D[i, 0], D[i, 1]  # Point angles (theta, phi)

            # Spherical law of cosines
            cos_angle = (np.sin(phi1) * np.sin(phi2) +
                         np.cos(phi1) * np.cos(phi2) * np.cos(theta1 - theta2))
            distances[i] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return distances
    
    def getH(a, D):
        tc_preftheta = a[0] % (2.*np.pi)
        tc_prefphi = a[1] % (2.*np.pi)
        tc_beta = a[2]
        tc_h = a[3]
        tc_sigma = np.pi/2
        if len(a) > 4:
            tc_sigma = a[4]
        distsqrds = ToroidalDist([tc_preftheta, tc_prefphi], D)
        return tc_h + tc_beta * np.exp(-distsqrds / (2*tc_sigma**2))

    def getLambdaOneCircle(params,D):
        tc_angle = params[0]% (2.*np.pi)
        tc_beta = params[1]
        tc_h = params[2]
        tc_sigma = params[3]
        dist = CircleDist(tc_angle, D)
        bump = tc_h + tc_beta * np.exp(-dist**2 / (2*(tc_sigma)**2))
        return bump

    def getLambdaTwoCircles(a, D):
        tc_preftheta = a[0] % (2.*np.pi)
        tc_prefphi = a[1] % (2.*np.pi)
        tc_beta = a[2]
        tc_h = a[3]
        #tc_choose = a[4]
        tc_sigma = np.pi/4
        if len(a) > 4:
           tc_sigma = a[4]
        dist1 = CircleDist(tc_preftheta, D[:, 0])
        dist2 = CircleDist(tc_prefphi, D[:, 1])
        bump1 = tc_beta * np.exp(-dist1**2 / (2*(tc_sigma)**2))
        bump2 = tc_beta * np.exp(-dist2**2 / (2*(tc_sigma)**2))
        bump2d = tc_h + bump1 + bump2 
        #print(max(bump2d))
        return bump2d      

    def logLtc(a, Sj, D):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            H = getLambdaTwoCircles(a, D)
            result = -np.sum(Sj * H - np.exp(H) - gammaln(Sj + 1))
            for warning in w:
                print(f"Warning caught in logLtc: {warning.message}, with meanvalue for H: {np.mean(H)}, and variable: {a}")
            return result
    def compute_loglj(params, estDir, S, i, j):
        H = getLambdaTwoCircles(params[j, :], estDir)
        loglj = np.sum(S[j, :] * H - np.exp(H) - gammaln(S[j, :] + 1))
        return i, j, loglj


    def getscorepp(Sj, estDir):



        def objFunc(a): # I minimize a loglikelihood(with a minus sign). Positive value here
                        # Therefore have to have a minus in front of penalty, since it is positive, and i want it large
            return logLtc(a, Sj, estDir)

        # Initial bounds and parameters
        bds = ((None,None), (None,None),(0,None), (None,None))
        a = np.zeros(4)
        fun = 10**10 

        # Phase 1: Optimize for all different values
        for theta in np.arange(0, 2*np.pi, np.pi/5.):
            for phi in np.arange(0, 2*np.pi, np.pi/5.):
                result = minimize(objFunc, [theta, phi, 1, 0], method='L-BFGS-B', bounds=bds, options={"maxiter":1})
                #result = minimize(objFuncPenalty, [theta, phi, 3, 0, 0.5], method='L-BFGS-B', bounds=bds, options={"maxiter":1})
                if (result.fun < fun):
                    fun = result.fun
                    a = result.x

        # Add the variance
        bds = ((None,None), (None,None), (0, None),(None,None), (0,None))
        result = minimize(objFunc, [a[0],a[1], a[2],a[3],np.pi/2], method='L-BFGS-B', bounds=bds, options={"maxiter":1})
        #result = minimize(objFuncPenalty, [a[0],a[1],a[2],a[3],a[4],np.pi/2], method='L-BFGS-B', bounds=bds, options={"maxiter":1})

        #print(result.x)
        return result
    

    # def meanandstd(points): #Output should be mean and standard deviation. 
    #     cpoints = np.zeros(2)
    #     cpoints[0] = np.arctan2(np.sum(np.sin(points[:, 0])), np.sum(np.cos(points[:, 0])))
    #     cpoints[1] = np.arctan2(np.sum(np.sin(points[:, 1])), np.sum(np.cos(points[:, 1])))
    #     def mindist(a):
    #         #return np.mean(ToroidalDist(a, points))
    #         dist1 = CircleDist(a[0], points[:, 0])
    #         dist2 = CircleDist(a[1], points[:, 1])
    #         return np.mean(dist1) + np.mean(dist2)
    #     res = minimize(mindist, cpoints + 0.01, method='L-BFGS-B')
    #     print("res.x: ",res.x, "\n \n \n res.fun: ", res.fun)
    #     response = (res.x, np.sqrt(res.fun))
    #     return response
    
    # def meanandstd_circles(points):
    #     mean_theta = circmean(points[:,0])
    #     mean_phi = circmean(points[:,1])
    #     var_theta = circvar(points[:,0])
    #     var_phi = circvar(points[:,1])
    #     std_theta = np.sqrt(var_theta)
    #     std_phi = np.sqrt(var_phi)
    #     mean_result = np.array([mean_theta, mean_phi])
    #     std_result = np.array([std_theta, std_phi])

    #     std_result = (std_theta + std_phi) / 2
    #     return mean_result, std_result

    # def meanandstd_circles(points):
    #     cpoints = np.zeros(2)
    #     cpoints[0] = np.arctan2(np.sum(np.sin(points[:, 0])), np.sum(np.cos(points[:, 0])))
    #     cpoints[1] = np.arctan2(np.sum(np.sin(points[:, 1])), np.sum(np.cos(points[:, 1])))
    #     def mindist(a):
    #         dist1 = CircleDist(a[0], points[:, 0])
    #         dist2 = CircleDist(a[1], points[:, 1])
    #         return np.mean(dist1) + np.mean(dist2)
    #     res = minimize(mindist, cpoints + 0.01, method='L-BFGS-B')
    #     response = (res.x, np.sqrt(res.fun))
    #     return response
    


    def meanandstd_one_circle(a):
        mean_theta = circmean(a)
        var_theta = circvar(a)
        #print(mean_theta, np.sqrt(var_theta))
        return mean_theta, np.sqrt(var_theta)



    def main_function(iteration):
        global updatingPenaltyWeight
        
        name = 'two_circles'
        with gzip.open('twocirclesseparable.pkl.gz', 'rb') as f:
            data = pickle.load(f)

        hd_sim = data['hd_train']
        S = np.array(data['S_train']) #neuron data
        output_file.write("shapes of S and hd_sim" + str( S.shape) + str( hd_sim.shape) + '\n')
        #output_file.write("average value of S: " + str(np.mean(S)) + '\n')
        T= len(S[0,:])
        N = len(S[:,0])
        print("Nlength: ", N, "\n TLength: ", T)
        print("S shape = ",S.shape)
        output_file.write("T length: "+ str(T)  + '\n' + "N length: "+ str(N)+ '\n \n')
        output_file.flush()

        
        def getpreddir(params, S=S, P=80, NX=3):
            T = S.shape[1]  # Number of time steps
            N = S.shape[0]
            def getlikelihood(Dtry):
                logl = np.zeros(T)
                for i in range(N):  # Iterate over trials
                    H = getLambdaTwoCircles(params[i, :], Dtry)
                    logl += (S[i, :] * H - np.exp(H) - gammaln(S[i, :] + 1))
                return np.exp(logl)

            particles = np.random.uniform(0, 20 * np.pi, size=(P, T, 2)) % (2. * np.pi)
            Dc = np.zeros((T, 2))
            Dcstd = np.zeros((T,2))

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


        if True:
            name = '%s_rand' % name
            estDir = (np.random.rand(T, 2) * 100 * 2. * np.pi) % (2 * np.pi)
            #estDir = np.transpose(hd_sim)%(2*np.pi) + np.random.normal(0,np.pi/2, size = (T,2))

        llvals = np.zeros(30)  # LLvals = Log Likelihood VALueS
        toleranseCount = 0     # toleranse counter for how many iterations without improvement
          #When do we turn on penalty which desides ensemble

        #Setting values for variables to be changed:
        bestLlvals = -np.inf


        

        for i in range(len(llvals)):
            tt = time.time()
            results_list = []

            print("\n ---------------  \n")

            with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
                results = list(executor.map(getscorepp, S, [estDir] * N))
            for result in results:
                results_list.append(result)
            params = np.array([res.x for res in results_list])
            print(params[-1])

            output_file.write('Time for maximization: ' + str((time.time() - tt)) + "\n")
            output_file.flush()
            
########################### PARALLELLo:

            with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
                futures = []
                for j in range(N):
                    futures.append(executor.submit(compute_loglj, params, estDir, S, i, j))

            for future in concurrent.futures.as_completed(futures):
                i, j, loglj = future.result()
                llvals[i] += loglj

###############################
            # for j in range(N):
            #     llvals[i] -= logLtc(params[j, :], S[j, :], estDir) - gammaln(S[j, :] + 1).sum()
            print("\n New iteration with: \n likelihood: ", llvals[i], '\n')

            if (i>=1):
                print("Best earlier likelihood: ", bestLlvals, '\n')
                print("how big is the change in percentage? ", -(llvals[i]- bestLlvals)/(bestLlvals),'\n')
                
            if (abs(llvals[i]- bestLlvals) < 500 and (i>1)):  # test if each iteration improves the likelihood
                print(" \n This iteration was considered bad :) \n")
                toleranseCount += 1         
            else:
                toleranseCount = 0

            # if (toleranseCount >= 3):               # If three iterations in a row does not improve, we have converged
            #     print("Here the iterations broke in the toleranseCount>=3 check \n")          
            #     break



            if (llvals[i] > bestLlvals):
                bestLlvals = llvals[i]

            output_file.write('\n \n At iteration %d, the total negative log-likelihood is %f' % (i, llvals[i]) + "\n")
            params_str = ', '.join(map(str, params[-1]))
            # output_file.write('The parameters(theta, phi, height, background, choose, sigma)= ' + params_str)
            output_file.flush()
            tt = time.time()

            estDir, estDirStd = getpreddir(params)

            data = {'params': params, 'llvals': llvals[i]}
            #data = {"estDir": estDir, 'params': params, 'llvals': llvals[i]}
            with gzip.open('%s_estDir.pkl.gz' % (name + str(i) + "_start_" + str(iteration)), 'wb') as f:
                pickle.dump(data, f)
     
            # if llvals[-1] == np.min(llvals):
            #     perStartBestData = estDir, estDirStd, totnegLogL, params, llvals[-1]

            output_file.write('Time for expectations' + str(time.time() - tt) + "\n")
            output_file.flush()

#Try and not update the values with more neurons:

        estDir, estDirStd = getpreddir(params)
        totnegLogL = 0
        for j in range(N):
            totnegLogL += logLtc(params[j, :], S[j, :], estDir) - gammaln(S[j,:]+1).sum()
        output_file.write("totnegLogL after recalculation= " + str(totnegLogL) + '\n \n')
        return estDir, estDirStd, totnegLogL, params, llvals
    

    if __name__ == "__main__":
        totnegLogL = -np.inf
        for i in range(10):
            np.random.seed(i * 100)
            estDir_new, estDirStd_new, totnegLogL_new, params_new, llvals_new = main_function(i)
            if abs(totnegLogL_new) < abs(totnegLogL):
                estDir, estDirStd, totnegLogL, params, llvals = estDir_new, estDirStd_new, totnegLogL_new, params_new, llvals_new

        data = {"estDir": estDir, "estDirStd": estDirStd, 'totnegLogL': totnegLogL, 'params': params, 'llvals': llvals}
        with gzip.open('best_estDir.pkl.gz', 'wb') as f:
            pickle.dump(data, f)
