"""
consists of functions to get a and c weighting information 
(DIN EN 61672-1) in different forms:
as weighting values for a given frequency numpy array
as a weighting vector (numpy array) for typical fft proecessing
as a weighting vector (numpy array) for a third-octave filterbank
as a weighting vector (numpy array) for an octave filterbank
as the complex transfer function
as filter coefficients (to be used by lfilter)
"""
# Author: J. Bitzer @ Jade Hochschule (copyright owner)
# License: BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause)
# Date: 12.12.2021
# used sources: DIN 61672-1, octave toolbox by Christophe COUVREUR
# version 1.0  12.12.2021 first build 
# 1.1 14.12.2021 added lower() to allow A and a 


import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

def __get_frequency_constants():
    """helper function with mathematical definition of the design frequencies (f1 to f4) for a and c-weighting curves"""
    fr = 1000 # section 5.4.6 in DIN 61672
    fl = 10**1.5 # section 5.4.6 in DIN 61672
    fh = 10**3.9 # section 5.4.6 in DIN 61672
    D = math.sqrt(1/2) # section 5.4.6 in DIN 61672
    b = (1/(1-D))*(fr**2 + fl**2 * fh**2/fr**2 - D*(fl**2 + fh**2) ) # eq 11 in DIN 61672
    c = fl**2 * fh**2 # eq 12 in DIN 61672
    f1 = ((-b - math.sqrt(b**2 - 4*c))*0.5)**0.5 # eq 9 in DIN 61672
    f4 = ((-b + math.sqrt(b**2 - 4*c))*0.5)**0.5 # eq 10 in DIN 61672
    fa = 10**2.45
    f2 = ((3-math.sqrt(5))*0.5)*fa # eq 13 in DIN 61672
    f3 = ((3+math.sqrt(5))*0.5)*fa # eq 14 in DIN 61672
    return f1,f2,f3,f4

def transfermagnitude2minimumphase(magH):
    '''Computes the minimum phase by a given magnitude of the transfer function by using the Hilbert transform'''    
    second_half = magH[-1:0:-1] 
    transfer_function = np.append(magH, second_half)
    tf_length = len(transfer_function)
    hilbert_in = np.log(np.abs(transfer_function))

    hilbert_out = -sp.hilbert(hilbert_in)
    phase = hilbert_out[:tf_length//2+1].imag 
    return phase

def FDLS_design(order_of_numerator, order_of_denominator, mag_h, phase_h, freq_vek, fs = -1):
    
    if (len(freq_vek) <= 1):
        omega = np.linspace(start = 0, stop = np.pi, num = len(mag_h))
    else:
        omega = 2*freq_vek/fs*np.pi
            
    y = mag_h*np.cos(phase_h); 
    X_D = -mag_h * np.cos(-1 * omega + phase_h); 

    X_D = [X_D]
    for k in range(order_of_denominator-1): 
        X_D.append(-mag_h * np.cos(-(k+2) * omega + phase_h))
    X_D = np.array(X_D).T 

    # non recursive part
    X_N = np.cos(-1 * omega);  

    X_N = [X_N]
    for k in range(order_of_numerator-1):
        X_N.append(np.cos(-(k+2) * omega))    
    X_N = np.array(X_N).T

    #and define X as input matrix
    X = np.hstack([X_D, np.ones([len(mag_h), 1]), X_N])
    coeff = np.linalg.lstsq(X, y, rcond=None)
    a = [1, *coeff[0][:order_of_denominator]] # ANN: einfach eine Liste statt hstack (* fügt Elemente ein, anstatt die ganze Liste)
    b = coeff[0][order_of_denominator:order_of_denominator+order_of_numerator+1]
    return b,a
    

def get_complex_tf_weighting(f_Hz, weight_func = 'a'):
    """returns the complex transfer function for a given frequency vector f_Hz"""
    f1,f2,f3,f4 = __get_frequency_constants()
    
    om_vek = 2*np.pi*1j*f_Hz
    
    if weight_func.lower() == 'c':
        c1000 = -0.062
        cweight = ((4*np.pi**2*f4**2*om_vek**2)/((om_vek + 2*np.pi*f1)**2 *(om_vek + 2*np.pi*f4)**2))/(10**(c1000/20))
        return cweight
    
    if weight_func.lower() == 'a':
        a1000 = -2 
        aweight = ((4*np.pi**2*f4**2 * om_vek**4)/((om_vek + 2*np.pi*f1)**2 *(om_vek + 2*np.pi*f2)*
                                                   (om_vek + 2*np.pi*f3)*(om_vek + 2*np.pi*f4)**2))/(10**(a1000/20))

        return aweight
    

def get_weight_value(f_Hz, weight_func = 'a', return_mode = 'log'):
    """returns the weighting values for a given frequency vector f_Hz in dB(log) or linear, specified by return_mode (default = 'log')"""
    f_Hz[f_Hz == 0] = 0.1 # prevent division by zero
    f1,f2,f3,f4 = __get_frequency_constants()   
    if weight_func.lower() == 'c':
        c1000 = -0.062
        # eq 6 in DIN 61672
        if return_mode == 'log':
            cweight = 20*np.log10((f4**2*f_Hz**2)/((f_Hz**2 + f1**2)*(f_Hz**2 + f4**2))) - c1000
        else:
            cweight = ((f4**2*f_Hz**2)/((f_Hz**2 + f1**2)*(f_Hz**2 + f4**2)))/(10**(c1000/20))
        return cweight
    
    if weight_func.lower() == 'a':
        a1000 = -2 
        if return_mode == 'log':
            # eq 7 in DIN 61672
            aweight = 20*np.log10((f4**2*f_Hz**4)/((f_Hz**2 + f1**2)*(f_Hz**2 + f2**2)**0.5*(f_Hz**2 + f3**2)**0.5*(f_Hz**2 + f4**2))) - a1000
        else:
            aweight = ((f4**2*f_Hz**4)/((f_Hz**2 + f1**2)*(f_Hz**2 + f2**2)**0.5*(f_Hz**2 + f3**2)**0.5*(f_Hz**2 + f4**2)))/(10**(a1000/20))
        return aweight
    
def get_fftweight_vector(fft_size, fs, weight_func = 'a', return_mode = 'log'):
    """ for a given fft_size the a or c weighting vector (fft_size/2 +1 elements from 0 to fs/2 Hz) is returned"""
    freq_vek = np.linspace(0, fs/2, num = int(fft_size/2+1))
    return get_weight_value(freq_vek, weight_func, return_mode), freq_vek

def get_onethirdweight_vector(weight_func = 'a'):
    """ the weights and frequencies for a 34 band one-third filterbank (startfreq = 10, endfreq = 20k) are returned"""
    a, freq_vek = get_spec(weight_func)
    return get_weight_value(freq_vek, weight_func), freq_vek

def get_octaveweight_vector(weight_func = 'a'):
    """the weights and frequencies for a 11 band octave filterbank (startfreq = 16, endfreq = 16k) are returned """
    freq_vek = np.array([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    return get_weight_value(freq_vek, weight_func), freq_vek

    
def get_weight_coefficients(fs, weight_func = 'a'):
    """designs an IIR filter that implements the desired weighting function for a given sampling rate fs 
    (fs > 40 kHz to fullfill class1 specification )
    design idea is based on the matlab octave toolbox. better solutions for lower fs should be possible
    see EOF for some ideas"""

    if fs < 40000:
        print("WARNING: the resulting filter coefficients will not fullfill the class1 constraint")

    f1,f2,f3,f4 = __get_frequency_constants()    
    den = np.convolve([1., +4*np.pi * f4, (2*np.pi * f4)**2], [1., +4*np.pi * f1, (2*np.pi * f1)**2])
    if weight_func.lower() == 'c' :
        c1000 = -0.062
        num = [(2*np.pi*f4)**2 * (10**(-c1000 / 20.0)), 0., 0.]
    
    if weight_func.lower() == 'a':
        a1000 = -2 
        den = np.convolve(np.convolve(den, [1., 2*np.pi * f3]),[1., 2*np.pi * f2])
        num = [(2*np.pi*f4)**2 * (10**(-a1000 / 20.0)), 0., 0., 0., 0.]
    
    b,a = sp.bilinear(num, den, fs)

    return b,a

    
def get_spec(weight_func = 'a'):
    """ returns the specification of a- and c-weighting function and the class1 limits
    parameter:
        weight_func:  'a' (default) or 'c'
    returns:
        tf_spec: a 34 element numpy-array  the specification for the transfer function of the weighting curve in dB
        f_reference: the corresponding vector with the reference frequencies
        class_1_upperlimit: corresponding vector of the allowed deviation in dB (positive)
        class_1_lowerlimit: corresponding vector of the allowed deviation in dB (negative dB, so you had to add this vector to tf_spec). 
    """
    # all values from table 2 in  DIN 61672
    fref = np.array([10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 
                     1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 ])
    class1_upper_limit = np.array([3.5, 3.0,2.5,2.5, 2.5,2.0,1.5, 1.5,1.5,1.5, 1.5,1.5,1.5, 1.5,1.4,1.4, 1.4,1.4,1.4, 
                                     1.4,1.1,1.4, 1.6,1.6,1.6, 1.6,1.6,2.1, 2.1,2.1,2.6, 3,3.5,4 ])
    class1_lower_limit = np.array([-np.inf, -np.inf,-4.5,-2.5, -2,-2,-1.5, -1.5,-1.5,-1.5, -1.5,-1.5,-1.5,  -1.5,-1.4,-1.4, 
                                   -1.4,-1.4,-1.4, -1.4,-1.1,-1.4, -1.6,-1.6,-1.6, -1.6,-1.6,-2.1, -2.6,-3.1,-3.6, -6.0,-17.0,-np.inf  ])
    if (weight_func.lower() == 'a'):
        a_desired = np.array([-70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, 
                              -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -.8, 0, .6, 1, 1.2, 1.3, 1.2, 1, .5, -.1, -1.1, -2.5, -4.3, -6.6, -9.3 ])
        return a_desired, fref, class1_lower_limit,  class1_upper_limit

    if (weight_func.lower() == 'c'):
        c_desired = np.array([ -14.3, -11.2, -8.5, -6.2, -4.4, -3.0, -2.0, -1.3, -.8, -.5, -.3, -.2, -.1, 0, 0, 0, 0, 0, 0, 
                              0, 0, 0, -.1, -.2, -.3, -.5, -.8, -1.3, -2.0, -3.0, -4.4, -6.2, -8.5, -11.2 ])
        return c_desired, fref, class1_lower_limit,  class1_upper_limit

if __name__ == '__main__':
    print('called as script')
    weight_type  = 'A'
    # print(get_weight_value(np.array([10,100,1000]),weight_type))
    a, f, ll, hl = get_spec(weight_type)
    fig,ax = plt.subplots()
    ax.plot(np.log(f),a, 'r') 
    ax.plot(np.log(f),a+hl,'g:')
    ax.plot(np.log(f),a+ll, 'g:') 
    fs = 48000
    afft,f_fft = get_fftweight_vector(1024,fs, weight_type)
    ax.plot(np.log(f_fft),afft, 'y+') 
    b,a = get_weight_coefficients(fs,weight_type)
    #print (b)
    #print (a)
    w, H = sp.freqz(b, a, worN =2048, fs = fs)
    ax.plot(np.log(w[1:]),20*np.log10(np.abs(H[1:])),'k')
    ax.set_ylim([-70.0, 5.0])
    ax.set_xlim([np.log(10), np.log(20000)])
    ax.set_xticks(np.log(f[:-1:4]))
    ax.set_xticklabels((f[:-1:4]))
    plt.show()

    

""" 
an idea to design the a and c filter by FDLS. it does not work, 
another better solution would be, wo use SOS filter and correct the tf at Nyquist.
next idea: use a better IIR arbitrary designb routine 

    #fft_size = 2*4096
    #tf,f = get_fftweight_vector(fft_size,fs,weight_func,'lin')
    #phase = transfermagnitude2minimumphase(np.sqrt(tf))
    
    # log weighting for LS Design
    #f_log = np.logspace(1.0,4.05, num = 150)
    f_log = np.linspace(100,10000, num = 200)
    # print(f_log)
    #index_f = (np.round( (2*f_log/fs*(fft_size/2+1)))).astype(int)
    #print(index_f)
    tf = get_complex_tf_weighting(f_log, weight_func)
    
    tf_mag = np.abs(tf)
    tf_phase = np.angle(tf)
    
    fig2,ax2 = plt.subplots(ncols=2)
    ax2[0].plot(f_log,20*np.log10(tf_mag))
    
    ax2[1].plot(f_log,np.unwrap(tf_phase))
    #plt.show()
   
    
    if weight_func == 'a':
        order_num = 8
        order_den = 8
    else:
        order_num = 4
        order_den = 4

    b,a = FDLS_design(order_num, order_den,tf_mag, np.unwrap(tf_phase), f_log, fs)
    
    return b,a
    
"""
     
"""Copyright <2021> <Joerg Bitzer>

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
     
"""