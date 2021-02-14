#!/usr/bin/env python3

def filter_f(Fs, Fpass1, Fpass2):
    # Fpass1 First Passband Frequency
    # Fpass2  Second Passband Frequency

    import numpy as np
    from scipy.signal import firls
    
    N = int(round(200*Fs/100))  # Order
    Fstop1 = int(Fpass1-2)   # First Stopband Frequency
    Fstop2 = int(Fpass2+2)  # Second Stopband Frequency
    Wstop1 = int(1)    # First Stopband Weight
    Wpass  = int(1)    # Passband Weight
    Wstop2 = int(1)    # Second Stopband Weight
    return firls(N+1, np.asarray([0, Fstop1, Fpass1, Fpass2, Fstop2, Fs/2])/(Fs/2), [0, 0, 1, 1, 0, 0],
                 [Wstop1, Wpass, Wstop2])
