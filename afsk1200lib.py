#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#File        pyrscwlib.py
#Author      Jonathan Rawlinson/M0ZJO
#Date        13/01/2019
#Desc.       This is the main software library for the CW decoding software "Pyrscw".
#            The original inspiration for this software was rscw.c (http://wwwhome.cs.utwente.nl/~ptdeboer/ham/rscw/) 
#            written by PA3FWM but the implemntation differs in a number of ways. This software was written to 
#            process audio recordings of the XW2 satellites but will work with any machine generated CW.


__author__ = "Jonathan/M0ZJO"
__copyright__ = "Jonathan/M0ZJO 2019"
__credits__ = ["PA3FWM for RSCW"]
__license__ = "MIT"
__version__ = "0.0.1"
__date__ = "05/01/2019"
__maintainer__ = "Jonathan/M0ZJO"
__status__ = "Development"

import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
import wavio
import datetime
import logging


logging.basicConfig(filename='PyAFSK1200.log', level=logging.INFO)

# Print info about the software
def print_header():
    name = """
.______   ____    ____ .______          _______.  ______ ____    __    ____ 
|   _  \  \   \  /   / |   _  \        /       | /      |\   \  /  \  /   / 
|  |_)  |  \   \/   /  |  |_)  |      |   (----`|  ,----' \   \/    \/   /  
|   ___/    \_    _/   |      /        \   \    |  |       \            /   
|  |          |  |     |  |\  \----.----)   |   |  `----.   \    /\    /    
| _|          |__|     | _| `._____|_______/     \______|    \__/  \__/     
"""
    print(name)
    
    print("### PyRSCW version %s release date %s ###" % (__version__, __date__))
    print("### Written by %s. Happy Beeping! ###\n\n" % __author__)

# Use wavio to load the wav file from GQRX
def open_wav_file(filename, resample = None):
    log("Opening: %s" % filename)
    wav_data = wavio.read(filename)
    log("Wavfile loaded. Len:%i, Fs:%iHz" % (wav_data.data.shape[0], wav_data.rate))
    # Resample down to internal rate for speed (we don't need massive amounts of bw)
    if resample != None:
        resampled_wav = resample_wav(wav_data.data[:,0], wav_data.rate, resample)
        wav_data.data = resampled_wav[0]
        wav_data.rate = resampled_wav[1]
        log("Wavfile resampled. Len:%i, Fs:%iHz" % (wav_data.data.shape[0], wav_data.rate))
    else:
        wav_data.data = wav_data.data[:,0]
    return wav_data

def resample_wav(wav, in_fs, out_fs):
    N = in_fs/out_fs
    if int(N) == N:
        # Integer rate
        if int(N) != 1:
            return signal.decimate(wav, int(N)), out_fs
        else:
            log("No decimation required (N = 1)")
            return wav, in_fs
    else:
        # Non Integer Rate
        log("Non-integer downsampling rates not supported")
        return wav, in_fs
        

# Plot section of file (for debugging purposes)
def plot_file(wav_data):
    plt.plot(wav_data.data[0:1000])
    plt.show()
    return

# Simple DC removal algorithem
def remove_dc(wav_data):
    wav_data.data = wav_data.data - np.mean(wav_data.data)
    return wav_data

# Generate PSD of file to work out the CF of the CW signal
def find_carrier(wav_data):
    f, Pxx_den = signal.welch(wav_data.data, wav_data.rate, nperseg=2048)
    
    max_value = np.argmax(Pxx_den)
    
    log("Carrier found at %0.4fHz" % f[max_value])
    
    carrier_freq = f[max_value]
    
    plot = False
    if plot:
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()
    
    return carrier_freq

# From the CF of the CW signal - generate I and Q carriers to downsample the signal to baseband
def generate_carriers(wav_data, carrier_freq):
    rate = wav_data.rate  # samples per second
    sample_length = len(wav_data.data)         # sample duration (seconds)
    f = carrier_freq     # sound frequency (Hz)
    t = np.linspace(0, sample_length/rate, sample_length, endpoint=False)
    sin_car = np.sin(2*np.pi * f * t)
    cos_car = np.cos(2*np.pi * f * t)
    log("Carriers Generated")
    return sin_car, cos_car

def butter_bandpass(lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

#def iir_bandpass(lowcut, highcut, fs, order=50):
    #offset = 100
    #nyq = 0.5 * fs
    #low = lowcut / nyq
    #high = highcut / nyq
    #b, a = signal.iirdesign([low, high],[low-offset, high+offset], 1, 40)
    #return b, a

# Design LPF to remove signal away from baseband
# TODO pull out filter design parms
def design_low_notch(fs, mode, plot = False):
    nyq_rate = fs/2
    
    cf = 1200
    offset = 25
    
    if mode == 1:
        # Help from -->> https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

        b, a = butter_bandpass(cf-offset, cf+offset, fs)
        
        log("Filter of order %i deisgned (low)" % len(b))
    
    if plot:
        w, h = signal.freqz(b, a, worN=8000)
        plt.plot((w/np.pi)*nyq_rate, 20 * np.log(np.absolute(h)), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        #ylim(-0.05, 1.05)
        plt.grid(True)
        plt.show()
        
    return b, a

def design_high_notch(fs, mode, plot = False):
    nyq_rate = fs/2
    
    cf = 2400
    offset = 25
    
    if mode == 1:
        # Help from -->> https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

        b, a = butter_bandpass(cf-offset, cf+offset, fs)
        
        log("Filter of order %i deisgned (high)" % len(b))
    
    if plot:
        w, h = signal.freqz(b, a, worN=8000)
        plt.plot((w/np.pi)*nyq_rate, 20 * np.log(np.absolute(h)), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        #ylim(-0.05, 1.05)
        plt.grid(True)
        plt.show()
        
    return b, a

# Downsample signal and apply LPF
def generate_filtered_baseband(wav_data):
    
    taps_high_b, taps_high_a = design_high_notch(wav_data.rate, 1, plot = False)
    taps_low_b, taps_low_a = design_low_notch(wav_data.rate, 1, plot = False)
    
    filtered_high = signal.filtfilt(taps_high_b, taps_high_a, wav_data.data)
    filtered_low = signal.filtfilt(taps_low_b, taps_low_a, wav_data.data)
    
    return [filtered_low, filtered_high]
    
# Compute magnitude of signal and normalise
def compute_abs(downsampled_power_data):
    sqr_dat = np.square(downsampled_power_data)
    sqr_dat = np.sqrt(sqr_dat)
    ## https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    #sqr_dat = sqr_dat / np.max(np.abs(sqr_dat))
    return sqr_dat/max(sqr_dat)

# Use a window to smooth the vector
# TODO Very inefficient...
def smooth_mag(mag, window_len = 10, No_iter = 1):
    for i in range(0, No_iter): 
        mag = np.convolve(mag, np.ones((window_len,))/window_len, mode='valid')
    return mag

# Convert +N -N to +1 -1
# TODO more efficient?
def quantise(low, high):
    
    diff = high - low
    
    bitstream_ind_pos = diff>0
    bitstream_ind_neg = diff<=0
    
    diff[bitstream_ind_pos] = 1
    diff[bitstream_ind_neg] = -1
    
    return diff

def get_start():
    start_str = "UOSAT-2"
    
             
# Generate alphabet (as defined by PA3FWM)
def generate_alphabet():
    
    alphabet = {}
    alphabet["A"] = "1000001"
    alphabet["B"] = "1000010"
    alphabet["C"] = "1000011"
    alphabet["D"] = "1000100"
    alphabet["E"] = "1000101"
    alphabet["F"] = "1000110"
    alphabet["G"] = "1000111"
    alphabet["H"] = "1001000"
    alphabet["I"] = "1001001"
    alphabet["J"] = "1001010"
    alphabet["K"] = "1001011"
    alphabet["L"] = "1001100"
    alphabet["M"] = "1001101"
    alphabet["N"] = "1001110"
    alphabet["O"] = "1001111"
    alphabet["P"] = "1010000"
    alphabet["Q"] = "1010001"
    alphabet["R"] = "1010010"
    alphabet["S"] = "1010011"
    alphabet["T"] = "1010100"
    alphabet["U"] = "1010101"
    alphabet["V"] = "1010110"
    alphabet["W"] = "1010111"
    alphabet["X"] = "1011000"
    alphabet["Y"] = "1011001"
    alphabet["Z"] = "1011010"
    alphabet["0"] = "0110000"
    alphabet["1"] = "0110001"
    alphabet["2"] = "0110010"
    alphabet["3"] = "0110011"
    alphabet["4"] = "0110100"
    alphabet["5"] = "0110101"
    alphabet["6"] = "0110110"
    alphabet["7"] = "0110111"
    alphabet["8"] = "0111000"
    alphabet["9"] = "0111001"
    alphabet["-"] = "0101101"
    
    
    return alphabet


# Convert a string of 1s and 0s to numpy array
def bin_string_to_numpy_array(bin_string):
    tmp = np.zeros(len(bin_string))
    for i in range(0, len(bin_string)):
        tmp[i] = int(bin_string[i])
    tmp[tmp == 0] = -1
    return tmp

# Main correlator function. Returns correlation value of two sequences.
def correlate_value(in1, in2):
    
    #plot_numpy_data([in1, in2])
    
    if len(in1) != len(in2):
        return -1
    out = np.zeros(len(in1))
    for i in range(0, len(in1)):
        out[i] = in1[i] * in2[i]
    
    return np.sum(out)/len(out)
 
# Decode bitstream block
# TODO pass through variables
def decode_block(bitstream, alphabet, wpm):
    
    ts = int(8000/1200) # In ms
    fs = 1000
    symbol_len = (fs * ts)/1000
    
    offset = 0
    tmp_str = ""
    while True:
        correlator_output = correlate_alphabet(bitstream[0 + offset:44*ts + offset], alphabet, ts)
        tmp_str = tmp_str + correlator_output[0]
        if correlator_output[1] == None:
            print("\n### Decode Complete ###")
            return tmp_str
        #print(correlator_output[0], end='', flush=True)
        
        offset = offset + correlator_output[1]

# Use correlator to search through all letters and numbers of alphabet dictionary. 
# Returns the following:
# - Letter (with or without space as appropriate)
# - Next sync value (to remove "bit jitter")
def correlate_alphabet(bits, alphabet, ts):
    alphabet_keys = list(alphabet.keys())
    alphabet_values = list(alphabet.values())
    
    #List for upsampled alphabet values
    alphabet_values_us = []
    # Upsample according to symbol len (will only fork for 1000Hz fs
    for i in range(0, len(alphabet_keys)):
        tmp_str = ""
        for j in range(0, len(alphabet_values[i])):
            tmp_str = tmp_str + repeat_to_length(alphabet_values[i][j], int(ts))
        alphabet_values_us.append(bin_string_to_numpy_array(tmp_str))
    
    # Somewhere to store the correlation output
    ans = np.zeros(len(alphabet_values_us))
    for i in range(0, len(alphabet_values_us)):
        ans[i] = correlate_value(alphabet_values_us[i], bits[0:len(alphabet_values_us[i])])
    
    # Find the index of the maxval
    correlator_result = np.argmax(ans)
        
    #plt.plot(bits)
    #plt.show()
    
    # Sync up to the next bit and detect space
    j = 0
    for offset_delta in range(len(alphabet_values_us[correlator_result]) - int(ts/2), len(bits)):
        if bits[offset_delta] == 1:
            if j < ts*3:
                return alphabet_keys[correlator_result], offset_delta
            else:
                return alphabet_keys[correlator_result] + " ", offset_delta
        j = j + 1
    
    # Next bit not found - signal must be over!
    return alphabet_keys[correlator_result], None

# From -->> https://stackoverflow.com/questions/3391076/repeat-string-to-certain-length
def repeat_to_length(string_to_expand, length):
    return string_to_expand*length

# Plot a series of numpy arrays (debugging...)
def plot_numpy_data(mag):
    
    for i in mag:
        plt.plot(i)
    plt.show()
    return

def log(string):
    print(datetime.datetime.now(), string)
    log_str = str(datetime.datetime.now()) + "\t" + string
    logging.info(log_str)
    
def output_data(string, work_id):
    print(string)
    with open("pyrscw_%s.txt" % work_id, "a+") as f:
        f.write(string + "\r\n")
    f.close()
    

# What if someone tries to run the library file!
if __name__ == "__main__":
    # execute only if run as a script
    print_header()
    print("This is the library file - please run the main script 'pyrscw.py'")
