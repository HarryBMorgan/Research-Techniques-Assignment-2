#-----------------------------------------------------------------------------
#PHY3054
#Assignment 2
#Student: 6463360
#Problem 2
#---Programme Description-----------------------------------------------------
#This programme generates a discrete sample from a probability density
#function using the transformation method. It calculates the CDF, F(x), from
#the PDF, f(x), and the ineverse F(x). The starting f(x) = Ax^alpha.
#This programme has been designed for easy changing of the variables and the
#number of alpha's to test, with no other code needing to be edited. A plot of
#each PDF alongside the data that resembles it is generated and saved.
#-----------------------------------------------------------------------------
#Import useful libraries.
import numpy as np
import matplotlib.pyplot as plt


#-Functions-------------------------------------------------------------------
#Define function for a calculation of A, f(x) and F(x)^-1.
#Some special cases have been programmed for alpha=-1 as division by zero
#would occur otherwise.
def calcA(alpha, xlo, xup): #Calculates value of constant A in PDF.
    if alpha == -1: #Special case for alpha=-1.
        return 1 / (np.log(abs(xup)) - np.log(abs(xlo)))
    else:   #All other values of alpha.
        return (alpha + 1) / (xup**(alpha + 1) - xlo**(alpha + 1))

def PDF(A, alpha, x):   #Function of PDF.
    return A * (x**alpha)

def CDF(A, alpha, x, xlo):
    if alpha == -1:
        return A * (np.log(x) - np.log(xlo))
    else:
        return A / (alpha + 1) * (x**(alpha + 1) - xlo**(alpha + 1))

def invCDF(A, alpha, xlo, R):    #Calculates inverse F(x).
    if alpha == -1: #Special case for alpha=-1.
        return np.exp((R / A) + np.log(abs(xlo)))
    else:   #All other values of alpha.
        return ((R * (alpha + 1) / A) + xlo**(alpha + 1))**(1 / (alpha + 1))


#-Main------------------------------------------------------------------------
#Set variables.
nbins = 100 #Number of bins to divide data into.
xlo, xup = 1, 5 #Set range of x range.
xp = np.linspace(xlo, xup, nbins)   #X points for parent distribution.

N = 10000    #Number of values in each sample.
np.random.seed(123) #Initialise the random number seed.
R = np.random.rand(N)   #Generate N random numbers.

alpha = [-2.35, -1.00] #List of alpha values to be used.
A = []  #Initialise A list for calculating any number of A values.
for i in range(len(alpha)): #Looped so more alpha values can be used easily.
    print("alpha%i =" %i, '%.2f' %alpha[i])
    A.append(calcA(alpha[i], xlo, xup))
    print("The value of A%i =" %i, '%.3f' %A[i])


#Check CDF is 0<F(x)<1 for xlo<x<xup.
for i in range(2):
    
    #Calculate CDF.
    F = CDF(A[i], alpha[i], xp, xlo)
    
    #Check it's in the correct range.
    if F[0] != 0 or F[-1] != 1:
        print("ERROR: The CDF is outside the correct range.")
        exit(1)


#Plot the Histograms.
for i in range(len(alpha)):
    
    #Create transformed data set.
    X = invCDF(A[i], alpha[i], xlo, R)
    X.sort()
    
    #Create bin list.
    logbins = np.geomspace(X.min(), X.max(), nbins) #Creats log_10 bins.
    binlist = [nbins, logbins]
    
    for j in range(2):  #Plot histogram for 2 bin types (linear and log_10).    
        
        #Plot the parent PDF.
        yp = PDF(A[i], alpha[i], xp)
        plt.plot(xp, yp, 'r--', linewidth=3, label="f(x) = {}x^{}" \
            .format('%.3f'%A[i], alpha[i]))
        
        #Plot histogram.
        height, bins, patches = plt.hist(X, bins=binlist[j], density=True, \
                                    color='g', label="Transformed Data Set")
        
        #Plot error bars assuming each bin is a Poisson process.
        hist, edges = np.histogram(X, bins=binlist[j])
        yerr = np.sqrt(hist) / hist
        yerr_xpos = 0.5 * (edges[1:] + edges[:-1])
        yerr_ypos = height
        plt.errorbar(yerr_xpos, height, yerr=yerr*height, color='k', \
            ecolor='k', fmt='.', capsize=3)
        
        #Format graph.
        if j == 1:  #Log_10 histogram formatting specifications.
            title = "Graph of Log10({}x^{})"
            xlabel, ylabel = "$Log_{10}(x)$", "$Log_{10}(f(x))$"
            plt.xscale('log'), plt.yscale('log')
        else:   #Linear histogram formatting specifications.
            title = "Graph of {}x^{}"
            xlabel, ylabel = "x", "f(x)"
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.xlabel(xlabel, fontsize=12), plt.ylabel(ylabel, fontsize=12)
        plt.xlim(edges[1], edges[-1])
        plt.title(title .format('%.3f' %A[i], alpha[i]))
        plt.legend(prop={'size': 12})
        plt.tight_layout()
        name = "Task 2 " + title + ".png"
        plt.savefig(name .format('%.3f' %A[i], alpha[i]))
        plt.show()