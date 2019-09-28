#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#File        afsk.....py
#Author      Jonathan Rawlinson/M0ZJO
#Date        13/01/2019
#Desc.       Desc.....

# Import util libs
import sys, getopt

help_txt = 'pyrscw.py -i <inputfile> -w <wpm> -o <work_id>'

def main(argv):
    inputfile = ''
    wpm = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","wpm=","work_id="])
    except getopt.GetoptError:
        print("Options error:")
        print(help_txt)
        sys.exit(2)
        
    if len(args) == 0:
        print("Options error:")
        print(help_txt)
        
    for opt, arg in opts:
        if opt == '-h':
            print(help_txt)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--work_id"):
            work_id = arg
    print("### Input file: ", inputfile)
    print("### Work ID is:", work_id)  
    
    # Import the DSP library
    import afsk1200lib
    from afsk1200lib import log

    # Set up variables
    min_signal_len = 2000 #In ms
    threshold_value = 0.1 #Normalised to varience of signal
    internal_resample_val = None #Internal fs for initial DSP (Reduce for speed)
    detection_threshold = 0.1 #Value to determine if a signal is present

    afsk1200lib.print_header()

    # Open file and load into array
    log("### Opening file ###")
    wav_file = afsk1200lib.open_wav_file(inputfile, resample = internal_resample_val)

    # Remove any DC present in the array.
    # TODO this should be a windowed method to run over the file
    log("### Remove DC ###")
    wav_file = afsk1200lib.remove_dc(wav_file)


    # Designs a FIR filter to remove non-CW components
    log("### Applying filters ###")
    filtered_data = afsk1200lib.generate_filtered_baseband(wav_file)
    
    #afsk1200lib.plot_numpy_data([filtered_data[0], filtered_data[1]])

    ## Dowsample full rate file to 1000Hz
    #log("### Downsampling ###")
    #downsampled_power_data = afsk1200lib.downsample_abs_bb(baseband_data, wav_file.rate)

    # Calculate the magnitude vector of the IQ
    log("### Calculate Mag ###")
    magnitude_data_low = afsk1200lib.compute_abs(filtered_data[0])
    magnitude_data_high = afsk1200lib.compute_abs(filtered_data[1])
    
    #afsk1200lib.plot_numpy_data([magnitude_data_low, magnitude_data_high])

    # Smooth the magnitude vector (convolutional method)
    log("### Smooth ###")
    magnitude_data_smoothed_low = afsk1200lib.smooth_mag(magnitude_data_low)
    magnitude_data_smoothed_high = afsk1200lib.smooth_mag(magnitude_data_high)
    
    afsk1200lib.plot_numpy_data([magnitude_data_smoothed_low, magnitude_data_smoothed_high])

    # Convert +n and -n to 1 and -1
    log("### Quantise ###")
    bitstream = afsk1200lib.quantise(magnitude_data_smoothed_low, magnitude_data_smoothed_high)
    
    afsk1200lib.plot_numpy_data([bitstream[1000000:1200000]])

    # From the signal detection vector - determine where the bt starts (to the nearest sample)
    log("### Bit Synch ###")
    signal_synch_list = afsk1200lib.bit_synch(bitstream, signal_present, min_length = min_signal_len)

    # Decode the synced bitstream
    log("### Decode ###")
    for i in range(0, len(signal_synch_list)):
        decoder_output = afsk1200lib.decode_block(bitstream[signal_synch_list[i][0]: signal_synch_list[i][1]], afsk1200lib.generate_alphabet(), wpm)
        afsk1200lib.output_data(decoder_output, work_id)

if __name__ == "__main__":
   main(sys.argv[1:])

