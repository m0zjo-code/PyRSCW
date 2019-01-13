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


logging.basicConfig(filename='PyRSCW.log', level=logging.INFO)

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

# Design LPF to remove signal away from baseband
# TODO pull out filter design parms
def design_lpf(fs, mode, plot = False):
    
    nyq_rate = fs/2
    
    # It is also possible to load in designs from other sources - not supported yet
    if mode == 0:
        with open("150hzlpf", "r") as f:
            taps = []
            for tap in f:
                taps.append(float(tap))
                
        log("Filter of order %i loaded" % len(taps))
    
    if mode == 1:
        # Help from -->> https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        width = 500/nyq_rate

        # The desired attenuation in the stop band, in dB.
        ripple_db = 40.0

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = signal.kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = 100.0

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
        
        log("Filter of order %i deisgned" % len(taps))
    
    if plot:
        w, h = signal.freqz(taps, worN=8000)
        plt.plot((w/np.pi)*nyq_rate, 20 * np.log(np.absolute(h)), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        #ylim(-0.05, 1.05)
        plt.grid(True)
        plt.show()
        
    return taps

# Downsample signal and apply LPF
def generate_filtered_baseband(wav_data, carriers):
    
    taps = design_lpf(wav_data.rate, 1)
    
    i_wave = carriers[0] * wav_data.data
    q_wave = carriers[1] * wav_data.data
    
    # Use lfilter to filter x with the FIR filter.
    filtered_i = signal.lfilter(taps, 1.0, i_wave)
    filtered_q = signal.lfilter(taps, 1.0, q_wave)
    
    return filtered_i, filtered_q

# Downsample audio file to 1000 Hz
# TODO pipe thorugh fs so different fs can be supported
def downsample_abs_bb(baseband_data, fs):
    i_chan = baseband_data[0]
    q_chan = baseband_data[1]
    
    N = int(fs/1000) # 48000/48 = 1000
    bb_len = len(i_chan)
    
    truncate_len = np.mod(bb_len, N)
    i_chan = i_chan[:bb_len-truncate_len]
    q_chan = q_chan[:bb_len-truncate_len]
    
    i_chan_ds = i_chan.reshape((int(len(i_chan)/N), N))
    q_chan_ds = q_chan.reshape((int(len(q_chan)/N), N))
    
    i_chan_ds = np.sum(i_chan_ds, axis = 1)
    q_chan_ds = np.sum(q_chan_ds, axis = 1)
    
    return i_chan_ds, q_chan_ds
    
# Compute magnitude of signal and normalise
def compute_abs(downsampled_power_data):
    sqr_dat = np.square(downsampled_power_data[0]) + np.square(downsampled_power_data[1])
    sqr_dat = np.sqrt(sqr_dat)
    ## https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    #sqr_dat = sqr_dat / np.max(np.abs(sqr_dat))
    return sqr_dat/max(sqr_dat)

# Use a window to smooth the vector
# TODO Very inefficient...
def smooth_mag(mag, window_len = 10, No_iter = 20):
    for i in range(0, No_iter):
        mag = np.convolve(mag, np.ones((window_len,))/window_len, mode='valid')
    return mag

# Determine the value of the power threshold
def get_threshold_val(mag, location, thresh = 0.5, search_len = 1000):
    #Define search area
    min_search = location-search_len/2
    if min_search < 0:
        min_search = 0
        
    max_search = location+search_len/2
    if max_search > len(mag) - 1:
        max_search = len(mag) - 1
    
    # "Cut out" vector to be searched and return max and min
    #search_vector = mag[int(min_search):int(max_search)]
    min_val = np.min(mag[int(min_search):int(max_search)])
    max_val = np.max(mag[int(min_search):int(max_search)])
    
    delta = max_val - min_val
    
    # Convert rel value to abs value
    threshold_val = min_val + delta*thresh
    
    return threshold_val

# Iterate through vector and compute threshold
# TODO can be made a lot more efficient by only sampling every N samples (as the value will move rather slowly... I think... )
def get_threshold(mag, thresh = 0.5):
    thresh_vector = np.zeros(mag.shape)
    for i in range(0, len(mag)):
        thresh_vector[i] = get_threshold_val(mag, i)
    
    return thresh_vector

# Determine the coefficient for signal detection
def signal_detect_val(mag, location, thresh = 0.5, search_len = 1000):
    
    min_search = location-search_len/2
    if min_search < 0:
        min_search = 0
        
    max_search = location+search_len/2
    if max_search > len(mag) - 1:
        max_search = len(mag) - 1

    search_vector = mag[int(min_search):int(max_search)]
    
    return np.std(search_vector)
    
# Iterate through vector and compute threshold
# TODO work out method of computing the detection_threshold (fixed atm). Probably something to do with the fs and noise of sound card (so can probably be assumed)
def signal_detect(mag, detection_threshold = 0.1):
    
    signal_detect_vector = np.zeros(mag.shape)
    for i in range(0, len(mag)):
        signal_detect_vector[i] = signal_detect_val(mag, i)
    
    signal_detect_vector = (signal_detect_vector > detection_threshold)
    
    return signal_detect_vector

# Apply the power threshold
def apply_threshold(mag, thresh_vector):
    return mag - thresh_vector

# Convert +N -N to +1 -1
# TODO more efficient?
def quantise(mag):
    
    bitstream_ind_pos = mag>0
    bitstream_ind_neg = mag<=0
    
    mag[bitstream_ind_pos] = 1
    mag[bitstream_ind_neg] = -1
    
    return mag

# Finds the start of each bitstream when a signal is present
def bit_synch(bitstream, signal_present, min_length = 1000):
    signal_len = len(bitstream)
    
    # Simple FSM to determine the bit start.
    SIGNAL_STATUS = False
    SYNCH = False
    
    signal_list = []
    
    # Check to see if we start off with signal already!
    if signal_present[0] == 1:
        SIGNAL_STATUS = True
    
    for i in range(1, signal_len):
        if (signal_present[i] == 1) and (signal_present[i-1] == 0):
            #Start of signal (signal detector)
            SIGNAL_STATUS = True
        if (bitstream[i] == 1) and (bitstream[i-1] == -1) and (SIGNAL_STATUS == True) and (SYNCH == False):
            #Start of signal (Start of bit)
            start_pos = i
            SYNCH = True
        if (signal_present[i] == 0) and (signal_present[i-1] == 1) and (SYNCH == True):
            #End of signal (bit)
            end_pos = i
            if (end_pos - start_pos) > min_length:
                signal_list.append([start_pos, end_pos])
            SIGNAL_STATUS = False
            SYNCH = False
    # The signal ended whilst we were synched! Make sure not to lose any data.
    if SYNCH == True:
        end_pos = signal_len - 1
        #print(end_pos - start_pos)
        if (end_pos - start_pos) > min_length:
            signal_list.append([start_pos, end_pos])
        
    return signal_list
             
# Generate alphabet (as defined by PA3FWM)
def generate_alphabet():
    
    alphabet = {}
    alphabet["a"] = "10111000"
    alphabet["b"] = "111010101000"
    alphabet["c"] = "11101011101000"
    alphabet["d"] = "1110101000"
    alphabet["e"] = "1000"
    alphabet["f"] = "101011101000"
    alphabet["g"] = "111011101000"
    alphabet["h"] = "1010101000"
    alphabet["i"] = "101000"
    alphabet["j"] = "1011101110111000"
    alphabet["k"] = "111010111000"
    alphabet["l"] = "101110101000"
    alphabet["m"] = "1110111000"
    alphabet["n"] = "11101000"
    alphabet["o"] = "11101110111000"
    alphabet["p"] = "10111011101000"
    alphabet["q"] = "1110111010111000"
    alphabet["r"] = "1011101000"
    alphabet["s"] = "10101000"
    alphabet["t"] = "111000"
    alphabet["u"] = "1010111000"
    alphabet["v"] = "101010111000"
    alphabet["w"] = "101110111000"
    alphabet["x"] = "11101010111000"
    alphabet["y"] = "1110101110111000"
    alphabet["z"] = "11101110101000"
    alphabet["0"] = "1110111011101110111000"
    alphabet["1"] = "10111011101110111000"
    alphabet["2"] = "101011101110111000"
    alphabet["3"] = "1010101110111000"
    alphabet["4"] = "10101010111000"
    alphabet["5"] = "101010101000"
    alphabet["6"] = "11101010101000"
    alphabet["7"] = "1110111010101000"
    alphabet["8"] = "111011101110101000"
    alphabet["9"] = "11101110111011101000"
    
    return alphabet
        
#def wpm_to_baud_rate(wpm):
    ## From -->> https://en.wikipedia.org/wiki/Words_per_minute
    #return float((50/60) * wpm)

# Convert WPM to something useful!
def wpm_to_symbol_len(wpm):
    # from https://www.eham.net/ehamforum/smf/index.php?topic=8534.0;wap2
    return 1200/wpm

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
    #bitstream = np.load("bitstream.npy")
    
    ts = int(1200/wpm) # In ms
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
