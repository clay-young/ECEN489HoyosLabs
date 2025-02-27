#Digital Filters ECEN 489 Data Conversion Lab 1''

#Importing Necessary Libraries into python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, tf2zpk

#FIR Filter Coefficients (b is the numerator, a is the denominator)
#So H(z) = (b[0]+b[1]z^(-1)+...+b[n]z^(-n))/(a[0]+a[1]z^(-1)+...+a[n]z^(-n))

b_fir = [1, 5, 6]
a_fir = [1]

#IIR Filter Coefficients
b_iir = [1]
a_iir = [1, 4]

#Function to plot frequency reponse and pole-zero plot of filters
def plot_filter_response(b, a, title):
    #Frequnecy response
    w, h = freqz(b, a, worN=8000);
    
    #Plotting magnitude response
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(w / np.pi, 20*np.log10(abs(h)), 'b')
    plt.title(f"Frequency Response - {title}")
    plt.xlabel('Normalized Frequnecy (xπ rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.grid() #Adding a grid to plot
    
    #Computing the plot pole-zero diagram
    z, p, k = tf2zpk(b,a)
    plt.subplot(2,1,2)
    #Create scatter plot
    plt.scatter(np.real(z), np.imag(z), s=100, marker='o', label='Zeros')
    plt.scatter(np.real(p), np.imag(p), s=100, marker='x', label='Poles')
    plt.title(f"Pole-Zero Plot - {title}")
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid()
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

#Plot FIR Filter
plot_filter_response(b_fir, a_fir, "FIR Filter")

#Plot IIR Filter
plot_filter_response(b_iir, a_iir, "IIR Filter")