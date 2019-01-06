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
from pyrscwlib import log

work_id = "XW2Data"

# Set up variables
# TODO this will be integrated into command args soon
wpm = 22
min_signal_len = 2000 #In ms
threshold_value = 0.5 #Normalised to varience of signal

pyrscwlib.print_header()

# Open file and load into array
log("### Opening file ###")
wav_file = pyrscwlib.open_wav_file("gqrx_20181230_160324_145727285.wav")

# Remove any DC present in the array.
# TODO this should be a windowed method to run over the file
log("### Remove DC ###")
wav_file = pyrscwlib.remove_dc(wav_file)

# Locate cw carrier (single for now)
# TODO - Compute this for the file in sections as the satellite tracking algorithem will not be perfect! 
log("### Find Carrier ###")
carrier_freq = pyrscwlib.find_carrier(wav_file)

# Generate baseband shifting carriers
# TODO implement the changing freq (change from above!)
log("### Generate Carriers ###")
carriers = pyrscwlib.generate_carriers(wav_file, carrier_freq)

# Designs a FIR filter to remove non-CW components
log("### Applying carriers and filter ###")
baseband_data = pyrscwlib.generate_filtered_baseband(wav_file, carriers)

# Dowsample full rate file to 1000Hz
log("### Downsampling ###")
downsampled_power_data = pyrscwlib.downsample_abs_bb(baseband_data, wav_file.rate)

# Calculate the magnitude vector of the IQ
log("### Calculate Mag ###")
magnitude_data = pyrscwlib.compute_abs(downsampled_power_data)

# Smooth the magnitude vector (convolutional method)
log("### Smooth ###")
magnitude_data_smoothed = pyrscwlib.smooth_mag(magnitude_data)

# Generate threshold vector to determine the level at which a signal is present
log("### Generate Threshold Vector ###")
thresh_vector = pyrscwlib.get_threshold(magnitude_data_smoothed, thresh = threshold_value)

# Use the threshold vector to return a zero mean signal
log("### Apply Threshold ###")
shifted_mag = pyrscwlib.apply_threshold(magnitude_data_smoothed, thresh_vector)

# Detect if a CW signal is present (or any signal for that matter) by looking at the standard deviation
log("### Detect Signal ###")
signal_present = pyrscwlib.signal_detect(magnitude_data, detection_threshold = 0.1)

#pyrscwlib.plot_numpy_data([magnitude_data, magnitude_data_smoothed, shifted_mag, signal_present])

# Convert +n and -n to 1 and -1
log("### Quantise ###")
bitstream = pyrscwlib.quantise(shifted_mag)

# From the signal detection vector - determine where the bt starts (to the nearest sample)
log("### Bit Synch ###")
signal_synch_list = pyrscwlib.bit_synch(bitstream, signal_present, min_length = min_signal_len)

# Decode the synced bitstream
log("### Decode ###")
for i in range(0, len(signal_synch_list)):
    decoder_output = pyrscwlib.decode_block(bitstream[signal_synch_list[i][0]: signal_synch_list[i][1]], pyrscwlib.generate_alphabet(), wpm)
    pyrscwlib.output_data(decoder_output, work_id)



