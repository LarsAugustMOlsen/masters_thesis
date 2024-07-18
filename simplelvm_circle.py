import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.ndimage
import sys
from scipy.optimize import minimize
from scipy.special import gammaln
import time
import pickle
import gzip


name = 'twocirclesseparable'
with gzip.open('%s.pkl.gz'%name, 'rb') as f:
  data = pickle.load(f)
name = '%s_%07d'%(name, np.floor(np.random.rand()*10**7))

hd_sim = np.ravel(np.array(data['hd_train']))
hd_sim = hd_sim % (2.*np.pi)

hd_sim_test = np.ravel(np.array(data['hd_test']))
hd_sim_test = hd_sim_test % (2.*np.pi)

S =np.array(data['S_train'])
print("S shape: ",S.shape)

N = len(S[:,0])
T = len(S[0,:])
print("N and T: ",N,T)

S_test = np.array(data['S_test'])

print("S_test shape: ", S_test.shape)
T_test = len(S_test[0,:])
print("N and T_test: ", N, T_test)

#####################
def getH(a, D): ## H is the "lambda" for the Poisson random variable
  tc_prefdir = a[0] % (2.*np.pi)
  tc_beta = a[1]
  tc_h = a[2]
  tc_sigma = np.pi/2
  if(len(a)>3):
    tc_sigma = a[3]
  dtheta = np.abs(D - tc_prefdir)
  dtheta = np.minimum(dtheta, 2.*np.pi - dtheta)
  return( tc_h + tc_beta * np.exp(-dtheta**2 / (2 * tc_sigma**2)) )

def logLtc(a, Sj, D): ## negative log likelihood of the Poisson (except for log factorial term)
  H = getH(a, D)
  return(-np.sum(Sj*H - np.exp(H)))

def getparams(DD):
    def getscore(j):
      def doone(a):
        return(logLtc(a, S[j,:], DD))
      bds = ((None, None), (0, None), (None, None))
      a = np.zeros(3)
      fun = 10**10
      for ang in np.arange(0, 2*np.pi, np.pi/4.):
        result = minimize(doone, [ang, 3, -2], method='L-BFGS-B', bounds=bds)
        if(result.fun<fun):
          fun = result.fun
          a = result.x
      bds = ((None, None), (0, None), (None, None), (np.pi/8, 4*np.pi))
      result = minimize(doone, [a[0], a[1], a[2], np.pi/2] , method='L-BFGS-B', bounds=bds)
      return(result)

    totnegLogL = 0
    params = np.zeros((N, 4))
    for i in range(N):
      res = getscore(i)
      params[i,:] = res.x
      totnegLogL += res.fun + np.sum(gammaln(S[i,:] + 1))
    return(params, totnegLogL)



def getpreddir(params, S1 = S,  NP=200, NX=2):
    T=len(S1[0,:])
    def getlikelihood(Dtry):
      logl = np.zeros(T)
      for i in range(N):
        H = getH(params[i,:], Dtry)
        logl += S1[i,:]*H - np.exp(H) - gammaln(S1[i,:]+1)
      return(np.exp(logl))

    P = NP
    particles = np.random.uniform(0, 20*np.pi, size=(P,T))%(2.*np.pi)
    Dc = np.zeros(T)
    Dcstd = np.zeros(T)

    ## use particle filter approach to find angles
    for xxx in range(NX): ## lets run it a few times... (NX)
        particles += np.random.normal(0, np.pi/4, size=(P,T)) ## HYPERPARAMETER!!!
        particles %= (2*np.pi)

        likelihoods = np.zeros((P,T))
        for i in range(P):
            Dt = particles[i,:]
            likelihoods[i,:] = getlikelihood(Dt)
        normfact = np.sum(likelihoods, axis=0)
        weights = likelihoods / normfact[np.newaxis, :]

        for t in range(T):
            inds = np.random.choice(np.arange(P), size=P, p=weights[:,t])
            particles[:,t] = particles[inds,t]
            angs = particles[:,t] + 0.
            if(xxx == NX-1):
              Dc[t] = np.arctan2(np.sum(np.sin(angs)), np.sum(np.cos(angs)))
              angs -= Dc[t]
              angs[angs<-np.pi] += 2*np.pi
              angs[angs> np.pi] -= 2*np.pi
              Dcstd[t] = np.std(angs)
    return(Dc, Dcstd)

##################


DD = (np.random.rand(T)*100)%(2*np.pi)
if(np.sum(np.isnan(DD))>0):
  DD[np.isnan(DD)] = 0.

### run for a long time (or check for convergence)
for i in range(10): 
  params, totlogL = getparams(DD)
  DD, DDstd = getpreddir(params)

  totnegLogL = 0
  for j in range(N):
      totnegLogL += logLtc(params[j,:], S[j,:], DD) + np.sum(gammaln(S[j,:] + 1))
  print(i, 'total negative log likelihood', totnegLogL)

  if(True):
    plt.figure(10, figsize=(5,5))
    plt.clf()
    plt.plot(hd_sim[0::2], DD, '.', ms=0.5, color='black')
    #plt.title('Total negative log(L) %f'%totnegLogL)
    plt.xlabel('true values')
    plt.ylabel('decoded values')
    plt.savefig('%s_dirs_%03d.png'%(name, i))

  


## NOW PLOTTING!!
def rotateanglestoalign(DDtry, hd_sim = hd_sim):
  hd_sim=hd_sim[0::2]
  def disttotrue(a):
    dd = np.abs(DDtry - hd_sim+a[0])
    dd = np.minimum(dd, 2.*np.pi - dd)
    return(np.sum(dd**2))

  #first try without flipping
  result = minimize(disttotrue, [np.pi], method='L-BFGS-B')
  val = result.fun
  offset = result.x[0]

  ## now try flipped
  DDorig = DDtry+0.
  DDtry = 2.*np.pi-DDtry
  result = minimize(disttotrue, [np.pi], method='L-BFGS-B')
  if(result.fun < val):
    return( (DDtry+result.x[0])%(2.*np.pi) )
  else:
    return( (DDorig+offset)%(2.*np.pi) )
    

### now use more particles and iterations to get nice stdevs estimates and plot them
DD, DDstd = getpreddir(params,S,  1000, 3)
DD = rotateanglestoalign(DD)
params, totlogL = getparams(DD)
print('Final negative log likelihood', totlogL)




# Make tuning curves
# angles = np.arange(0, 2.*np.pi+0.0001, np.pi/10)
# anglevertices = (0.5*(angles[0:(-1)]+angles[1:]))
# tuningcurveO = np.zeros((N, len(angles)-1))
# tuningcurveD = np.zeros((N, len(angles)-1))
# for i in range(len(angles)-1):
#   indsO = (hd_sim>=angles[i]) * (hd_sim<angles[i+1])
#   indsD = (DD>=angles[i]) * (DD<angles[i+1])
#   for j in range(N):
#     tuningcurveO[j,i] = np.mean(S[j,indsO])
#     tuningcurveD[j,i] = np.mean(S[j,indsD])
# tuningcurveF = np.zeros((N, len(anglevertices)))
# for i in range(N):
#   tuningcurveF[i] = np.exp(getH(params[i,:], anglevertices))

# fig = plt.figure(1, figsize=(10,10))
# for i in range(N):
#   fr = plt.subplot(4,4,i+1)
#   plt.plot(anglevertices, tuningcurveO[i,:], '-', color='orange')
#   plt.plot(anglevertices, tuningcurveD[i,:], '-', color='purple')
#   #plt.plot(anglevertices, tuningcurveF[i,:], '-', color='green')
#   plt.ylim(0, 1.1*np.max([np.max(tuningcurveO[i,:]), np.max(tuningcurveD[i,:]), np.max(tuningcurveF[i,:])]))
#   fr.axes.get_xaxis().set_visible(False)
# fig.suptitle('Orange are observed and purple decoded tuning curves')
# plt.savefig('%s_tuningcurves.png'%(name))


DD_test, DDstd_test = getpreddir(params, S_test)

DD_test = rotateanglestoalign(DD_test, hd_sim_test)
# Plot the true vs. decoded values for the test data
plt.figure(10, figsize=(5,5))
plt.clf()
plt.plot(hd_sim_test[0::2], DD_test, '.', ms=0.5, color='black')
#plt.title('Total negative log(L) %f'%totlogL)
plt.xlabel('true values')
plt.ylabel('decoded values')
plt.savefig('%s_dirs_test.png'%(name))
plt.show()


DD_plot = rotateanglestoalign(DD)
plt.figure(10, figsize=(5,5))
plt.clf()
plt.plot(hd_sim[0::2], DD_plot, '.', ms=0.5, color='black')
plt.xlabel('true values')
plt.ylabel('decoded values')
plt.savefig('final_angles_decoding.png')


data = {'DD':DD, 'params':params, 'totlogL':totlogL}
with gzip.open('best_estDir.pkl.gz', 'wb') as f:
    pickle.dump(data, f)
