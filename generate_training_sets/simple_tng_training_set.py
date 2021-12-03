import os
import numpy as np

import astropy

import astropy.io.ascii as ascii

import tng_api_utils as tau


if __name__=="__main__":
    print("generating a simple training set from TNG API...")

    #collect pristine images

    #specify simulation parameters
    sim='TNG100-1'
    mstar_range=[1.0e9,None]

    #identify subhalos -- use catalogs extracted from JupyterLab
    input_cat_file = 'input_catalogs/tng100_snap40_mstar9.0.txt'

    input_data = ascii.read(input_cat_file)
    print(input_data)


    #create "mock observed" versions


    #collect into useful data structures for ML trianing
