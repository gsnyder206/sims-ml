import os
import numpy as np

import astropy

import astropy.io.ascii as ascii
import astropy.io.fits as fits
from astropy.table import Table

import tng_api_utils as tau
from tqdm import tqdm


if __name__=="__main__":
    print("generating a simple training set from TNG API...")



    #specify simulation parameters
    sim='TNG100-1'
    mstar_range=[1.0e9,None]

    #identify subhalos -- use catalogs extracted from JupyterLab
    input_cat_file = 'input_catalogs/tng100_snap40_mstar9.0.txt'
    snapgroup = 'snap40'

    input_data = ascii.read(input_cat_file)
    print(input_data[0:5])

    target_npix = 128
    target_pixsize_arcsec = 0.06

    fwhm_arcsec = 0.1
    #filter1='f814w'
    #filter1='f200w'
    filter1='f356w'
    if filter1=='f200w' or filter1=='f356w' or filter1=='f444w':
        filstring='jwst_'+filter1
    elif filter1=='f814w':
        filstring='wfc_acs_'+filter1

    sigma_njy = 0.01

    #store as a handful of HDUs per FITS file?
    #then combine?

    file_storage_dir='/astro/snyder_lab/MockSurveys/sims-ml-data'

    #start_i=0
    #stop_i=10

    #do everything
    start_i=0
    stop_i=len(input_data)-1

    N_images=stop_i - start_i

    data_array_pristine = np.zeros(shape=(target_npix,target_npix,N_images))
    data_array_fwhm = np.zeros(shape=(target_npix,target_npix,N_images))
    data_array_noise = np.zeros(shape=(target_npix,target_npix,N_images))

    input_catalog = input_data[start_i:stop_i]

    fake_phdu = fits.PrimaryHDU()

    fake_phdu.header['SOURCE']=input_cat_file

    input_table_hdu = fits.BinTableHDU(input_catalog)
    input_table_hdu.header['EXTNAME']='InputCatalog'
    input_table_hdu.header['SOURCE']=input_cat_file


    output_dict = {'input_index':np.zeros(shape=(N_images),dtype=np.int64),
                    'orig_index':np.zeros(shape=(N_images),dtype=np.int64),
                    'status':np.zeros(shape=(N_images),dtype=np.int32),
                    'ABmag':np.zeros(shape=(N_images)),
                    'Snap':np.zeros(shape=(N_images),dtype=np.int32),
                    'SubfindID':np.zeros(shape=(N_images),dtype=np.int32)}


    for i,row in enumerate(tqdm(input_catalog,mininterval=1,miniters=10,smoothing=0.1)):
        #figure out size to get

        mstar_halfrad_kpc = row['HalfRadMstar']
        redshift = row['redshift']

        kpc_per_arcmin = tau.tngcos.kpc_proper_per_arcmin(redshift).value

        halfrad_arcmin = mstar_halfrad_kpc/kpc_per_arcmin

        image_fov_arcmin = target_pixsize_arcsec*target_npix/60.0

        image_fov_kpc = image_fov_arcmin*kpc_per_arcmin

        image_fov_halfrad = image_fov_kpc/mstar_halfrad_kpc


        #collect pristine images

        lev=row['SubfindID']

        pristine_outfile1=os.path.join(file_storage_dir,sim,str(row['Snap']),filter1,'pristine_'+str(row['SubfindID'])+'.fits')

        try:
            pristine_hdu = tau.get_subhalo_mockdata_as_fits(sim=sim,snap=row['Snap'],sfid=row['SubfindID'],
                                        partType='stars',partField='stellarBandObsFrame-'+filstring,
                                        size=image_fov_arcmin,nPixels=target_npix,axes='0,1',existingheader=None)

            if i==0:
                image_header=pristine_hdu.header
                del image_header['SNAP']
                del image_header['SFID']


            data_array_pristine[:,:,i]=pristine_hdu.data
            #create "mock observed" versions

            mock_hdu = tau.convolve_with_fwhm(pristine_hdu, fwhm_arcsec=fwhm_arcsec)
            data_array_fwhm[:,:,i]=mock_hdu.data
            data_array_noise[:,:,i]=data_array_fwhm[:,:,i]+sigma_njy*np.random.randn(target_npix,target_npix)



            #prepare output summaries
            output_dict['status'][i]=1
            flux_njy=np.sum(pristine_hdu.data)
            if flux_njy > 0:
                output_dict['ABmag'][i]=-2.5*np.log10((1.0e-9)*flux_njy/3631.0)
            output_dict['Snap'][i]=row['Snap']
            output_dict['SubfindID'][i]=row['SubfindID']
            output_dict['input_index'][i]=i
            output_dict['orig_index'][i]=start_i+i

            #print(output_dict['ABmag'][i], output_dict['Snap'][i], output_dict['SubfindID'][i])

        except Exception as e:
            print('Error in this row, skipping...', row['Snap'], row['SubfindID'])
            print(e)
            continue

        #print(pristine_hdu.header.cards)




    dataset_hdu = fits.ImageHDU(data_array_pristine,header=image_header)
    dataset_hdu.header['SNAPGRP']=snapgroup

    fwhm_hdu = fits.ImageHDU(data_array_fwhm,header=dataset_hdu.header)
    fwhm_hdu.header['PSF_FWHM']=(fwhm_arcsec,'arcsec')
    fwhm_hdu.header['EXTNAME']=dataset_hdu.header['EXTNAME']+'PSF'


    noise_hdu = fits.ImageHDU(data_array_noise,header=fwhm_hdu.header)
    noise_hdu.header['RMS_NJY']=sigma_njy
    noise_hdu.header['EXTNAME']=dataset_hdu.header['EXTNAME']+'PSFNoise'


    output_table = Table(output_dict)
    outcat_hdu = fits.BinTableHDU(output_table)
    outcat_hdu.header['EXTNAME']='OutputCatalog'
    out_hdulist = fits.HDUList([fake_phdu,input_table_hdu,outcat_hdu,dataset_hdu,fwhm_hdu,noise_hdu])

    out_dir = os.path.join(file_storage_dir,'training_sets',sim,filter1,snapgroup,'run1')
    if not os.path.lexists(out_dir):
        os.makedirs(out_dir)

    out_hdulist.writeto(os.path.join(out_dir,'imagedata_'+snapgroup+'.fits'),overwrite=True)

    #collect into useful data structures for ML trianing
