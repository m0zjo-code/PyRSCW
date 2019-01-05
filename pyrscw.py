#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#File        pyrscw.py
#Author      Jonathan Rawlinson/M0ZJO
#Date        05/01/2019
#Desc.       This is the main script for the CW decoding software "Pyrscw".
#            The original inspiration for this software was rscw.c (http://wwwhome.cs.utwente.nl/~ptdeboer/ham/rscw/) 
#            written by PA3FWM but the implemntation differs in a number of ways. This software was written to 
#            process audio recordings of the XW2 satellites but will work with any machine generated CW.

# Import the DSP library
import pyrscwlib

# Set up variables
# TODO this will be integrated into command args soon
wpm = 22
min_signal_len = 2000 #In ms

# Open file and load into array
print("### Opening file ###")
wav_file = pyrscwlib.open_wav_file("gqrx_20181230_160324_145727285.wav")

# Remove any DC present in the array.
# TODO this should be a windowed method to run over the file
print("### Remove DC ###")
wav_file = pyrscwlib.remove_dc(wav_file)

# Locate cw carrier (single for now)
# TODO - Compute this for the file in sections as the satellite tracking algorithem will not be perfect! 
print("### Find Carrier ###")
carrier_freq = pyrscwlib.find_carrier(wav_file)

# Generate baseband shifting carriers
# TODO implement the changing freq (change from above!)
print("### Generate Carriers ###")
carriers = pyrscwlib.generate_carriers(wav_file, carrier_freq)

# Designs a FIR filter to remove non-CW components
print("### Applying carriers and filter ###")
baseband_data = pyrscwlib.generate_filtered_baseband(wav_file, carriers)

# Dowsample full rate file to 1000Hz
print("### Downsampling ###")
downsampled_power_data = pyrscwlib.downsample_abs_bb(baseband_data, wav_file.rate)

# Calculate the magnitude vector of the IQ
print("### Calculate Mag ###")
magnitude_data = pyrscwlib.compute_abs(downsampled_power_data)

# Smooth the magnitude vector (convolutional method)
print("### Smooth ###")
magnitude_data_smoothed = pyrscwlib.smooth_mag(magnitude_data)

# Generate threshold vector to determine the level at which a signal is present
print("### Generate Threshold Vector ###")
thresh_vector = pyrscwlib.get_threshold(magnitude_data_smoothed)

# Use the threshold vector to return a zero mean signal
print("### Apply Threshold ###")
shifted_mag = pyrscwlib.apply_threshold(magnitude_data_smoothed, thresh_vector)

# Detect if a CW signal is present (or any signal for that matter) by looking at the standard deviation
print("### Detect Signal ###")
signal_present = pyrscwlib.signal_detect(magnitude_data)

# Convert +n and -n to 1 and -1
print("### Quantise ###")
bitstream = pyrscwlib.quantise(shifted_mag)

# From the signal detection vector - determine where the bt starts (to the nearest sample)
print("### Bit Synch ###")
signal_synch_list = pyrscwlib.bit_synch(bitstream, signal_present, min_length = min_signal_len)

# Decode the synced bitstream
print("### Decode ###")
for i in range(0, len(signal_synch_list)):
    decoder_output = pyrscwlib.decode_block(bitstream[signal_synch_list[i][0]: signal_synch_list[i][1]], pyrscwlib.generate_alphabet(), wpm)
    print(decoder_output)



