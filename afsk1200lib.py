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
__credits__ = ["Surrey University"]
__license__ = "MIT"
__version__ = "0.0.1"
__date__ = "01/10/2019"
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
  _____                ______ _____ _  ____ ___   ___   ___  
 |  __ \         /\   |  ____/ ____| |/ /_ |__ \ / _ \ / _ \ 
 | |__) |   _   /  \  | |__ | (___ | ' / | |  ) | | | | | | |
 |  ___/ | | | / /\ \ |  __| \___ \|  <  | | / /| | | | | | |
 | |   | |_| |/ ____ \| |    ____) | . \ | |/ /_| |_| | |_| |
 |_|    \__, /_/    \_\_|   |_____/|_|\_\|_|____|\___/ \___/ 
         __/ |                                               
        |___/
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

def butter_bandpass(lowcut, highcut, fs, order=4):
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


def design_bp(fs, mode, plot = False):
    nyq_rate = fs/2
    
    cf = 1800
    offset = 900
    
    if mode == 1:
        # Help from -->> https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

        b, a = butter_bandpass(cf-offset, cf+offset, fs)
        
        log("Filter of order %i deisgned (high)" % len(b))
    
    if mode == 2:
        f1 = (cf-offset)/nyq_rate
        f2 = (cf+offset)/nyq_rate
        b = signal.firwin(200, [f1, f2], pass_zero=False)
        a = 1.0
    
    if plot:
        w, h = signal.freqz(b, a, worN=48000)
        plt.plot((w/np.pi)*nyq_rate, 20 * np.log(np.absolute(h)), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        #ylim(-0.05, 1.05)
        plt.grid(True)
        plt.show()
        
    return b, a

def design_lp(fs, mode, plot = False):
    nyq_rate = fs/2
    
    cf = 1200*1.2
    
    if mode == 2:
        b = signal.firwin(200, cf/nyq_rate)
        a = 1.0
    
    if plot:
        w, h = signal.freqz(b, a, worN=48000)
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
   
    # FIR (mode = 2) or IIR (mode = 1) Design
    taps_high_b, taps_high_a = design_high_notch(wav_data.rate, 2, plot = False)
    taps_low_b, taps_low_a = design_low_notch(wav_data.rate, 2, plot = False )
    taps_bandpass_wb_b, taps_bandpass_wb_a = design_bp(wav_data.rate, 2, plot = False)
    
    filtered_wav = signal.lfilter(taps_bandpass_wb_b, taps_bandpass_wb_a, wav_data.data)
    filtered_high = signal.lfilter(taps_high_b, taps_high_a, filtered_wav)
    filtered_low = signal.lfilter(taps_low_b, taps_low_a, filtered_wav)
    
    return [filtered_low, filtered_high]

def filter_discriminator(low, high, fs = 48000):
    b, a = design_lp(fs, 2, plot = False)
    
    filt_d = signal.lfilter(b, a, high-low)

    
    return filt_d
    
    

def fsk_demodulate(y, f_sep, cf, fs, baud):
    import time
    block_len = int(fs/baud)
    
    t = np.arange(0, block_len)/fs
    space_f = np.exp(-1j * 2 * np.pi * t * -f_sep/2)
    mark_f  = np.exp(-1j * 2 * np.pi * t *  f_sep/2)
    
    cos_carr = np.cos(2 * np.pi * cf * t)
    sin_carr = np.sin(2 * np.pi * cf * t)
    
    bitstream = []
    
    for i in range(0, len(y)-block_len, block_len):
        block = y[i:i+block_len]
        I_sig = sin_carr * block
        Q_sig = cos_carr * block
        block_cplx = I_sig + 1j * Q_sig
        
        
        space_b = block_cplx * space_f
        mark_b  = block_cplx * mark_f
        
        int_val_s = np.abs(np.sum(space_b))
        int_val_m = np.abs(np.sum(mark_b))
        
        if int_val_m > int_val_s:
            bit = 1
        else:
            bit = 0
        
        bitstream.append(bit)
        
    return bitstream

def PLL(NRZa, a = 0.74 , fs = 48000, baud = 1200):
    
    ctr_max = 2**20
    
    ctr = 0
    
    ctr_list = []
    
    idx = []
    
    for n in range(1, len(NRZa)-1):
        prev = NRZa[n-1]
        new = NRZa[n]
        
        if prev != new:
            ctr = ctr + a * (ctr_max*baud)/(fs)
        else:
            ctr = ctr + (ctr_max*baud)/(fs)
        
        if ctr>ctr_max:
            idx.append(n)
            ctr = 0
        ctr_list.append(ctr)
        #print(ctr, ctr_max-ctr)
    return idx, ctr_list/np.max(ctr_list)

def decode_block(bits):
    i = 0
    output_str = ""
    while True:
        if i > len(bits) - 12:
            break
        
        if [bits[i], bits[i+1], bits[i+10], bits[i+11]] == [1, 0, 1, 1]:
            #print(bits[i:i+12])
            data = np.array(bits[i+2:i+9])
            data_str = ""
            for k in range(0, 7):
                data_str = data_str + str(data[6-k])
            test_parity = bits[9]
            #print(test_parity, getParity(int(data_str, 2)))
            i = i + 11 
            output_str = output_str + return_char(data_str)
        else:
            i = i + 1
    return output_str
# https://www.geeksforgeeks.org/program-to-find-parity/
def getParity( n ): 
    parity = 0
    while n: 
        parity = ~parity 
        n = n & (n - 1) 
    return -parity 

def return_char(bits):
    inv_map = {v: k for k, v in generate_alphabet().items()}
    try:
        return inv_map[bits]
    except:
        return "."

def numpy_array_to_str(bits):
    out = ""
    for i in bits:
        if i == 1:
            out = out + "0"
        else:
            out = out + "1"
    return out
    
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
    #alphabet["CR"] = "0001101"
    #alphabet["LF"] = "0001010"
    alphabet["\r"] = "0001101"
    alphabet["\n"] = "0001010"
    alphabet["---Record Seperator---\n"] = "0011110"
    #alphabet["RS"] = "0011110"
    alphabet["US"] = "0011111"
    alphabet["NULL"] = "0000000"
    alphabet[" "] = "0100000"
    
    return alphabet

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
