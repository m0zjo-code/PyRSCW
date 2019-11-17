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
    internal_resample_val = None #Internal fs for initial DSP (Reduce for speed)

    afsk1200lib.print_header()

    # Open file and load into array
    log("### Opening file ###")
    wav_file = afsk1200lib.open_wav_file(inputfile, resample = internal_resample_val)

    # Remove any DC present in the array.
    log("### Remove DC ###")
    wav_file = afsk1200lib.remove_dc(wav_file)
    
    log("### Bandpass Filter ###")
    wav_file = afsk1200lib.filter_wav(wav_file)
    
    log("### Demodulate FSK ###")
    bitstream = afsk1200lib.fsk_demodulate(wav_file.data, 1200, 1800, wav_file.rate, 1200)
    
    log("### Decode UOSAT-2 Data ###")
    ascii_out = afsk1200lib.decode_block(bitstream)
    
    log("### Output UOSAT-2 Data to File ###")
    afsk1200lib.output_data(ascii_out, work_id)

if __name__ == "__main__":
   main(sys.argv[1:])

