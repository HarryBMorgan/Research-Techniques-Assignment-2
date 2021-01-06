#-----------------------------------------------------------------------------
#PHY3054
#Assignment 2
#Student: 6463360
#Problem 3
#---Programme Description-----------------------------------------------------
#This programme generates a random data set then calculates the lilelihood
#of a range of alpha values to fit the parent distribution. The uncertainity
#of this value is also calculated. The amount of values less than 1sigma away
#from the actual value is printed to scree. The range of possible alpha values
#are plotted as a histogram.
#-----------------------------------------------------------------------------
#Import useful libraries.
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo


#-Functions-------------------------------------------------------------------
#Define function for a calculation of A, F(x)^-1 and the negative 
#log-likelihood. Some special cases have been programmed for alpha=-1 as
#division by zero would occur otherwise.
def calcA(alpha, xlo, xup): #Calculates value of constant A in PDF.
    if alpha == -1: #Special case for alpha=-1.
        return 1 / (np.log(abs(xup)) - np.log(abs(xlo)))
    else:   #All other values of alpha.
        return (alpha + 1) / (xup**(alpha + 1) - xlo**(alpha + 1))

def invCDF(alpha, xlo, xup, R):    #Returns a transformed data set of x-values.
    if alpha == -1: #Special case for alpha=-1.
        return np.exp((R / calcA(alpha, xlo, xup)) + np.log(abs(xlo)))
    else:   #All other values of alpha.
        return ((R * (alpha + 1) / calcA(alpha, xlo, xup)) + \
                    xlo**(alpha + 1))**(1 / (alpha + 1))

def minus_loglike(alpha, args):    #Calculates the negative Log-Likelihood.
    x, xlo, xup = args
    return - sum(np.log(calcA(alpha, xlo, xup)) + alpha * np.log(x))

def asig(alpha, args):  #Function for calculating bisect to get asig.
    x, xlo, xup, maxM = args
    args = [x, xlo, xup]
    M = - minus_loglike(alpha, args)
    return M - maxM + 0.5   #Will equal 0 at bisect.


#-Main------------------------------------------------------------------------
#Set variables for getting random data set.
np.random.seed(123) #Initialise the random number seed.
N = 10000    #Number of values in each sample.
xlo, xup = 1, 5 #Set values of x range.
alpha = -2.35
runs = 1000 #Number of runs to do.


#Find minimum of minusM to get the best approximation for alpha with each 
#random data set. Do this 1000 times.
maxM, amax = [], []   #Fun and alpha list for minimize.
sigma = []  #Sigma list to contain max likely alpha uncertainities.
accept_a = []   #List of amax values withing 1 sig of actal alpha.
for i in range(runs):
    
    #Generate random data set of x-values.
    R = np.random.rand(N)
    x = invCDF(alpha, xlo, xup, R)
    x.sort()
    xlo2, xup2 = x[0], x[-1]
    args = [x, xlo2, xup2]    #Set arguments based on data.
    
    if i == 0:  #Print a graph of M on the first pass of the loop.
        
        #Find X limits xlo and xup and print.
        
        print("The dataset limits rounded to the nearest integer are:")
        print("xlo =", '%.5f' %xlo2)
        print("xup =", '%.5f' %xup2)
        
        #Calculate likelihood for range of alpha values.
        alpha_try = np.linspace(-4, 0, 100) #Alpha values to try.
        M = []
        for ialpha in alpha_try:
            M.append(-minus_loglike(ialpha, args))
        
        #Plot M graph.
        plt.plot(alpha_try, M)
        plt.title("Log-Likelihood of Alpha Values")
        
        plt.xlabel("alpha", fontsize=12), plt.ylabel("M", fontsize=12)
        plt.xlim(alpha_try[0], alpha_try[-1])   #Set axis limits.
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()  #Prevents cutting bits off graph when saved as png.
        plt.savefig("Task 3 Log-Likelihood Graph.png")
        plt.show()
    
    #Calculate most likely alpha value.
    res = spo.minimize(minus_loglike, alpha_try[0], args=((x, xlo2, xup2), ))
    maxM.append(-res.fun)  #Array of M for each alpha approximation.
    amax.append(res.x[0])  #Array of alpha value approxiations.
    
    #Calculate sigma's assuming M is Gaussian.
    args.append(maxM)
    a_sig = spo.bisect(asig, -10, amax[i], args=((x, xlo2, xup2, maxM[i]), ))
    sigma.append(a_sig - amax[i])
    
    #How many sigma is amax away from actual alpha?
    test = abs(amax[i] - alpha) / abs(sigma[i])
    if test <= 1.0: accept_a.append(amax[i])
    
    #Print statement to show programme running.
    print("Running...", '%.2f' %(i/runs*100), "%", end='\r')
print("Completed...", '%.2f' %(100), "%")


#Tell user how many alpha's are within 1 sig of -2.35.
print(len(accept_a)/len(amax)*100, "% of alphas calculated are within 1 sigma \
of alpha =", alpha)


#Plot to see all maxM values,
plt.hist(amax, bins=100)
hist, edges = np.histogram(amax, bins=100)
plt.title("Distribution of Most Likely Alpha Values")

plt.xlabel("alpha", fontsize=12), plt.ylabel("maxM", fontsize=12)
plt.xlim(edges[0], edges[-1])

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()  #Prevents cutting bits off graph when saved as png.
plt.savefig("Task 3 Maximum Alpha.png")
plt.show()