import numpy as np
#import matplotlib.pyplot as plt
import scipy.io
import scipy.ndimage
import sys
from scipy.optimize import minimize
from scipy.special import gammaln
import time
import concurrent.futures
import pickle
import gzip

output_file_path = 'output_parallel.txt'
with open(output_file_path, 'w') as output_file:
    

    #distance between two points on  torus
    def ToroidalDist(angles, D): #CHANGED
        D = np.array(D)
        theta = angles[0]
        phi = angles[1]
        #print("tehta, phi, D: ", theta, phi, D)
        
        dx = np.zeros_like(D[:,0])
        dy = np.zeros_like(D[:,0])
        
        for i in range(len(D)):
            dx[i] = abs(D[i, 0] - theta)
            dy[i] = abs(D[i, 1] - phi)
            dx[i] = np.minimum(dx[i], 2 * np.pi - dx[i])
            dy[i] = np.minimum(dy[i], 2 * np.pi - dy[i])
        #print(dx,dy)
        return np.sqrt(dx**2 + dy**2)


    def getH(a, D): ## H is the "lambda" for the Poisson random variable
        tc_preftheta = a[0] % (2.*np.pi) # Angles
        tc_prefphi = a[1] % (2.*np.pi)
        tc_beta = a[2] # This is a weight of the gaussian bumb
        tc_h = a[3] # This is background fire rate
        tc_sigma = np.pi/2 ### note, HYPERPARAMETER, also the width of the bumb or variance of the normal dist. 
        if(len(a)>4):
            tc_sigma = a[4]
        distsqrds = ToroidalDist([tc_preftheta, tc_prefphi], D)
        return( tc_h + tc_beta * np.exp(-distsqrds / (2*tc_sigma**2)) )

    def logLtc(a, Sj, D): ## negative log likelihood of the Poisson (except for log factorial term)
        H = getH(a, D)
        #print("sizes of return values from LogLtc: Sj, H =", Sj.shape, H.shape )
        return (-np.sum(Sj*H - np.exp(H)))


    def getparams(S, estDir):
        ### first find parameters for a binned (stupid) space
        T = len(estDir[:,0])
        nbins = 8 ## note len(a) = nbins^2
        XX = np.zeros((T, nbins*nbins))
        torbins = np.zeros(nbins*nbins)>10.
        angs = np.linspace(-0.000001, 2.*np.pi+0.000001, nbins+1)
        angcens = 0.5*(angs[1:]+angs[:(-1)])
        for i in range(nbins):
            whichA = (estDir[:,0]>angs[i])*(estDir[:,0]<=angs[i+1])
            for j in range(nbins):
                XX[:,i*nbins+j] = int(whichA*(estDir[:,1]>angs[j])*(estDir[:,1]<=angs[j+1]))
                if(np.sum(XX[:,i*nbins+j])>0):
                    torbins[i*nbins+j] = True

    ### for each neuron (in parallel)
    ### second find a single bump that best fits the binned (stupid) space
    ### third use those values as a starting point to find a local optimum


    def getscorepp(Sj, estDir):
        def doone(a):
            return(logLtc(a, Sj, estDir))
        bds = ((None, None), (None, None),  (0, None), (None, None))
        a = np.zeros(4)
        fun = 10**10 
        ## an alternative would be to first do simple bins, fit bump to that and then local optimization
        ## i should really switch to that, this is way too slow... also that way would probably do better
        for theta in np.arange(0, 2*np.pi, np.pi/5.):
            for phi in np.arange(0, 2*np.pi, np.pi/5.):
                #print(theta, phi)
                result = minimize(doone, [theta, phi, 3, -2], method='L-BFGS-B', bounds=bds)
                #print("getscorepp result from first minimizer",result)
                if(result.fun<fun):
                    fun = result.fun
                    a = result.x
        bds = ((None, None), (None, None), (0, None), (None, None), (np.pi/8, 2*np.pi))
        result = minimize(doone, [a[0], a[1], a[2], a[3], np.pi/2] , method='L-BFGS-B', bounds=bds)
        #print("getscprepp returns: ",result)
        return(result)


    def meanandstd(points):
        cpoints = np.zeros(2)
        cpoints[0] = np.arctan2(np.sum(np.sin(points[:,0])), np.sum(np.cos(points[:,0])))
        cpoints[1] = np.arctan2(np.sum(np.sin(points[:,1])), np.sum(np.cos(points[:,1])))
        def mindist(a):
            return(np.mean(ToroidalDist(a, points)))
        res = minimize(mindist, cpoints+0.01, method='L-BFGS-B')
        response = (res.x, np.sqrt(res.fun))
        #print("meanandstd response: ",response)
        return response
    
    #One log likelihood, test func for parallelization
    def compute_loglj(params, estDir, S, gammaln, i, j):
        H = getH(params[j, :], estDir)
        loglj = np.sum(S[j, :] * H - np.exp(H)) + np.sum(gammaln(S[j, :] + 1))
        return i, j, loglj

    

    #
    def main_function(iteration):
        name = 'justonecircle' #CHANGED
        with gzip.open('justonecircle.pkl.gz', 'rb') as f:
            data = pickle.load(f)

        # Extract data
        hd_sim = data['circvalues']
        S = np.array(data['S']) #neuron data
        output_file.write("shapes of S and hd_sim" + str( S.shape) + str( hd_sim.shape) + '\n')
        output_file.write("average value of S: " + str(np.mean(S)) + '\n')
        T= len(S[0,:])
        N = len(S[:,0])
        output_file.write("T length: "+ str(T)  + '\n' + "N length: "+ str(N))
        output_file.flush()
     
     
        startt = 0
        binsize = 20     #CHANGED        


        def getpreddir(params, P=200, NX=3):
            def getlikelihood(Dtry):
                logl = np.zeros(T)
                for i in range(N):
                    H = getH(params[i,:], Dtry)
                    logl += (S[i,:]*H - np.exp(H) - gammaln(S[i,:]+1))
                #logl= logl/N
                #output_file.write("np.exp(logl) returned from getlikelihoods: "+ str(np.exp(logl))+ "\n")
                return(np.exp(logl)) 

            
            T = S.shape[1]  # Number of time steps based on provided S
            particles = np.random.uniform(0, 20*np.pi, size=(P, T, 2)) % (2.*np.pi)
            Dc = np.zeros((T, 2))
            Dcstd = np.zeros(T)

            for xxx in range(NX):
                particles = particles + np.random.normal(0, np.pi/4, size=(P, T, 2))
                particles %= (2*np.pi)

                likelihoods = np.zeros((P, T))
                for i in range(P):
                    Dt = particles[i,:,:]
                    likelihoods[i,:] = getlikelihood(Dt)
                #output_file.write(str(np.exp(likelihoods)))
                #likelihoods = np.exp(likelihoods)
                normfact = np.sum(likelihoods, axis=0)
                #normfact[normfact == 0] = 1  # Avoid division by zero
                weights = likelihoods / normfact[np.newaxis, :]
      
                for t in range(T):
                    inds = np.random.choice(np.arange(P), size=P, p=weights[:,t])
                    particles[:,t,:] = particles[inds,t,:]
                    if(xxx == NX-1):  
                        angs = particles[:,t,:] + 0.
                        Dc[t,:], Dcstd[t] = meanandstd(angs)
            return(Dc, Dcstd)


        if(True):
            name = '%s_rand'%name
            output_file.write(name + "\n")
            np.random.seed()
            estDir = (np.random.rand(T,2)*100*2.*np.pi)%(2*np.pi) #make random start for estimated dir.

        llvals = np.zeros(20)
        for i in range(len(llvals)): 
            tt = time.time()
            # print("before start of parallel block: mean of estDir = ", np.mean(estDir),"\n")
            # print('dir x', estDir[:10,0])
            # print('dir y', estDir[:10,1])
            with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
                results = list(executor.map(getscorepp, S, [estDir]*N))
            #print("after parallel block: mean of S = ",np.mean(S), "\n")
            params = np.array([res.x for res in results])
            # print('params', params[0,:])

            output_file.write('Time for maximization: '+ str((time.time()-tt)) + "\n")
            output_file.flush()


            with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
                futures = []
                for j in range(N):
                    futures.append(executor.submit(compute_loglj, params, estDir, S, gammaln, i, j))

            for future in concurrent.futures.as_completed(futures):
                i, j, loglj = future.result()
                llvals[i] += loglj

        
            print('llvals', llvals[i])
            output_file.write('At iteration %d, the total negative log-likelihood is %f'%(i, llvals[i]) + "\n")
            output_file.flush()
            tt = time.time()
            data = {"estDir":estDir, 'params':params, 'llvals':llvals[i]}
            with gzip.open('%s_estDir_correctangle.pkl.gz'%(name + str(i)+ "_start_"+ str(iteration)), 'wb') as f:
                pickle.dump(data, f)
            output_file.write("params mean: "+ str(np.mean(params))+ "\n")
            estDir, estDirStd = getpreddir(params)
            # print('dir x', estDir[:10,0])
            # print('dir y', estDir[:10,1])
            # print(i, params[0,:])
            output_file.write("estDir mean and estDirStd: " + str(np.mean(estDir)) + str(np.mean(estDirStd))+ "\n")
            output_file.write('Time for expectations' + str(time.time()-tt) + "\n")
            output_file.flush()
            

        ### now use more particles and iterations to get nice stdevs estimates and plot them
        estDir, estDirStd = getpreddir(params, 1000, 3)
        totnegLogL = 0
        for j in range(N):
            totnegLogL += logLtc(params[j,:], S[j,:], estDir) + np.sum(gammaln(S[j,:] + 1))
        output_file.write("totnegLogL after recalculation= " + str(totnegLogL))
        return (estDir, estDirStd, totnegLogL, params, llvals)


    if __name__ == "__main__":
        totnegLogL = -np.inf
        for i in range(1):
            estDir_new, estDirStd_new, totnegLogL_new, params_new, llvals_new = main_function(i)
            if abs(totnegLogL_new) < abs(totnegLogL):
                estDir, estDirStd, totnegLogL, params, llvals = estDir_new, estDirStd_new, totnegLogL_new, params_new, llvals_new


        data = {"estDir":estDir, "estDirStd":estDirStd, 'totnegLogL':totnegLogL, 'params':params, 'llvals':llvals}
        with gzip.open('best_estDir_correctangle.pkl.gz', 'wb') as f:
            pickle.dump(data, f)