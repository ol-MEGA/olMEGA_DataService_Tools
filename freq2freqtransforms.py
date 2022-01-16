"""
This module implements functions to get transform matrix from fft sized spectra to fractional octave bands
"""
# Author: J. Bitzer @ Jade Hochschule (copyright owner)
# License: BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause)
# Date: 14.12.2021
# used sources: IEC 1260, 
# version 0.1  14.12.2021 init
# version 1.0 15.12 fractional octave is done

# to do
# mel and bark transforms https://www.dsprelated.com/freebooks/sasp/Bark_Frequency_Scale.html, 

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


def get_spectrum_fractionaloctave_transformmatrix(fft_size, fs, fstart = 10, fend = 20000, fractional_factor = 3, mode = 'linearinterp'):
    """
    computation of the transfrom matrx (fft_size/2 + 1 x nrofbands) for b-th fractional octave bands
    Input: 
        fft_size: the fft_size of the input spectrum for the transform. 
        fs: the sampling rate of the original data in Hz
        fstart: the lowest band to use (use nominal frequencies for standard b) default = 10
        fend: the highest band to use (use nominal frequencies) default = 20000
        fractional_factor: b in IEC 1260, typical values are 1 for octaves, 3 for on-third octaves and 
                            12 for semi-tone analysis (be aware for 12 the refrence frequency is 440 Hz not the usual 1k Hz)
                            default is 3
        mode: defines how the edges of the bands are treated default is 'linearinterp'. Use 'nearest' for 
                a resulting matrix with just ones in it. Especially at low frequencies and small fft_sice 
                the results will be wrong (far to high).
    Return:
        trans_mat: the resulting transformation matrix of size fft_size/2 + 1 x nr_of_bands
        f_mid: the computed mid frequencies
        f_nominal: for b = 3 and b = 1 the nominal frequencies given in IEC 1260 
                        + 10 Hz and 20kHz (for b = 3)
    """
    if fend > fs/2:
        fend = fs/2
    
    if fstart < 10:
        fstart = 10
        
    if fend > 20000:
        fend = 20000
    
    f_ref = 1000            
    index_faktor = (fft_size/2)/(fs/2)
    b = fractional_factor
    f_nominal = []
    if b == 12: # musical half-tone
        f_ref = 440
        band_nr = np.arange(-60,67) # 13.75 -- 19900 Hz if f_ref = 440
    elif b == 3: # one-third
        band_nr = np.arange(-21,15) # 2 more for bandedges
        f_nominal = np.array([10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 
                     1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 ])
        f_nominal = f_nominal[f_nominal >=fstart]
        f_nominal = f_nominal[f_nominal <=fend]
        
    elif b == 1:
        band_nr = np.arange(-6,5) # 2 more for bandedges
        f_nominal = np.array([16, 32, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        f_nominal = f_nominal[f_nominal >=fstart]
        f_nominal = f_nominal[f_nominal <=fend]
            
    f_mid = f_ref*2**((band_nr)/b)
    f_mid_low = f_mid*2**(1/(2*b))
    f_mid = f_mid[f_mid_low >= fstart]
    f_mid_high = f_mid*2**(-1/(2*b))
    f_mid = f_mid[f_mid_high <= fend]
    
    f_bandedges = np.append(f_mid[0]*2**(-1/(2*b)), f_mid*2**(1/(2*b)))
    f_bandedges[f_bandedges>fs/2] = fs/2
                
    nr_of_bands = int(len(f_mid))    
    trans_mat = np.zeros((int(fft_size/2+1),nr_of_bands))
    for kk in range(nr_of_bands):
        f_index_low = f_bandedges[kk]*index_faktor
        f_index_high = f_bandedges[kk+1]*index_faktor
        trans_mat[int(f_index_low):int(f_index_high)+1,kk] = 1
        if mode == 'linearinterp':
            trans_mat[int(f_index_high),kk] = f_index_high-int(f_index_high)
            trans_mat[int(f_index_low),kk] = 1-(f_index_low-int(f_index_low))
            if (int(f_index_high) == int(f_index_low)):
                trans_mat[int(f_index_high),kk] = f_index_high-f_index_low
    
    return trans_mat, f_mid, f_nominal
    


if __name__ == '__main__':
    """ 
    a simple test function to show the usage of get_spectrum_fractionaloctave_transformmatrix
    """
    fft_size = 1024
    fs = 44100
    fstart = 10
    fend = 20000
    fractional_factor = 3
    m, fmid, fnom = get_spectrum_fractionaloctave_transformmatrix(fft_size, fs, fstart = fstart, fend = fend, fractional_factor= fractional_factor)
    
    # white spectrum to test
    x = np.random.randn(fft_size)
    X = np.fft.fft(x)
    absX = np.abs(X[:int(fft_size/2 +1)])
    dispX = absX.dot(m)
    
    fig,ax = plt.subplots()
    if fractional_factor == 1 or fractional_factor == 3:
        ax.plot(fnom, dispX,'k+')
    else:
        ax.plot(fmid, dispX,'k+')
    
    plt.show()
    
    
"""
def get_spectrum_psychoacousticbands_transformmatrix():
    if band_def.lower() == 'bark':
        f_mid = np.array([50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500])
        f_nominal = f_mid
        f_bandedges = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500])
        
    elif band_def.lower() == 'mel':
        fref = 1000
"""    

    
"""Copyright <2021> <Joerg Bitzer>

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
     
"""