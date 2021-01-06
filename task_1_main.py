#-----------------------------------------------------------------------------
#PHY3054
#Assignment 2
#Student: 6463360
#Problem 1
#---Programme Description-----------------------------------------------------
#This programme uses Simpson's Rule of integration to calculate the value of
#the first order Bessel function.
#This is compared with Scipy's jv(v,z) function.
#The intensity of light from a point source with lambda=500nm over a radius
#of 0<=r<=1 micro metres. This is plotted.
#-----------------------------------------------------------------------------
#---Import useful libraries---
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi   #Gets the Simp's integration function.
import scipy.special as sps #Imports the jv(v,z) function.


#-Functions-------------------------------------------------------------------
#-Function of J_1(x)---
def J1(x):
    theta = np.linspace(0, np.pi, 1000) #Range of theta with N=1000.
    
    def func(x, theta): #Define function in the integration.
        return np.cos(theta - x * np.sin(theta))
    
    y = func(x, theta)  #Make y the array of func to be integrated over theta.
    
    return (1 / np.pi) * spi.simps(y, theta)  #Integrate over the theta range.


#-Main------------------------------------------------------------------------
#-Comparing my function with Scipy's---
#Compare the written J1(x) function with the values returned by Scipy's
#function jv(v,z) for Bessel functions of the first kind of order v=1 over a
#range od 0<=x<=20.

x = np.linspace(0, 20, 1000)    #Create x-range.

#Run J1(x).
myJ1 = np.empty(len(x))
for i in range(len(x)):
    myJ1[i] = J1(x[i])

spsJ1 = sps.jv(1, x)


#-Plot the results for comparrison---
plt.plot(x, myJ1, color='b', linestyle=':', linewidth=3, \
        label="My $J_1(x)$")
plt.plot(x, spsJ1, color='k', linestyle='-', linewidth=1, \
        label="Scipy's $J_1(x)$")
plt.title("My $J_1(x)$ Function vs Scipy's $J_1(x)$ Function")

plt.xlim(0, 20)
plt.xlabel("X", fontsize=12)
plt.ylabel("$J_1(x)$", fontsize=12)

ax = plt.gca()  #Alter the spines to make the graph nicer to look at.
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.legend(prop={'size': 12})
plt.tight_layout()
plt.savefig("Task 1 Bessel Comparrison Graph.png")
plt.show()


#-Computing intensity of diffraction---
#Calculating the value of I(r) for x=kr.

lam = 500e-9   #Units of m.
k = 2 * np.pi / lam
intensity = np.empty([500, 500])
step = 1e-6 / len(intensity)    #Step distance for 500 points over r=1microm.

for i in range(len(intensity)):
    
    X = step * i
    
    for j in range(len(intensity)):
    
        Y = step * j
        r = np.sqrt(X**2 + Y**2)
        R = k * r
        
        if r < 1e-9:  #Rule: as r->inf, I->0.5.
            intensity[i,j] = 0.5
        else:
            intensity[i,j] = (J1(R) / R)**2
    
    print("Running...", '%.2f' %(i/len(intensity)*100), "%", end='\r')
print("Completed...", '%.2f' %(100), "%")


#Create a full image of the diffraction pattern. Creating segments of the
#full image by rotating the origional intensity.
Intensity = np.empty([1000, 1000])  #Creating an array for all four quadrents.
Intensity[0:500, 0:500] = np.rot90(intensity, 2)    #Top left segment.
Intensity[0:500, 500:1000] = np.rot90(intensity, 1) #Top right segment.
Intensity[500:1000, 0:500] = np.rot90(intensity, 3)    #Bottom left segment.
Intensity[500:1000, 500:1000] = intensity.copy()  #Bottom right segment.

#Plot the diffraction image.
plt.imshow(Intensity, vmax=0.01)    #vmax prevents "glare" from the centre.
plt.title("Modlled Diffraction of a Light Source Viewed Through a Telescope")

lab = "Diameter of Focal Plane: Units of 0.002\u03BCm"
plt.xlabel(lab, fontsize=12), plt.ylabel(lab, fontsize=12)

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)

clb = plt.colorbar()
clb.set_label("Intensity", fontsize=12)

plt.gray()
plt.tight_layout()
plt.savefig("Task 1 Diffraction Pattern Figure.png")
plt.show()