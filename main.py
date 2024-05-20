import matplotlib.pyplot as plt
import scipy
import pyklip.klip as klip
from astropy.io import ascii
import pandas as pd
from scipy.stats import trim_mean
import warnings
from astropy.table import Table, Column, unique
import astropy.io.fits as fits
import numpy as np
import glob, os
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage.filters import uniform_filter
from datetime import date
import pyklip.instruments.NIRC2 as NIRC2
import pyklip.parallelized as parallelized


def undistort(inimage, x_dist, y_dist, fileName):
    # File paths for nirc2 distortion maps
    # Download files from the NIRC2 Distortion wiki: https://github.com/jluastro/nirc2_distortion/wiki
    """
    Non-PyRAF/IRAF version of nirc2dewarp.py -- note: this method does not conserve flux!
​
    Using distortion solution from Yelda et al. 2010, based on 'nirc2dewarp.py' found on NIRC2 Distortion wiki
​
    Files needed: Distortion x and y fits map (set path in distXgeoim and distYgeoim)
    (Download files from the NIRC2 Distortion wiki: https://github.com/jluastro/nirc2_distortion/wiki)
        nirc2_X_distortion.fits
        nirc2_Y_distortion.fits
    Input: inimage = (2D numpy array) distorted image
    Output: outimage = (2D numpy array) Undistorted image
​
    """
    imgsize = inimage.shape[0]



    ## if subarray, assume it's in the middle
    offset = (x_dist.shape[0] - imgsize) / 2
    print(offset)

    gridx, gridy = np.meshgrid(np.arange(imgsize, dtype=float), np.arange(imgsize, dtype=float))
    gridx -= x_dist
    gridy -= y_dist
    print(gridx, gridy)
    nanCount = count_nans_in_fits(fileName)

    outimage = klip.nan_map_coordinates_2d(inimage, gridy, gridx, {"order": 1})

    return (outimage)


def distortioncorrection(distXgeoim, distYgeoim, src):
    """
    Function for the distortion correction
    :param distXgeoim: Distortion X correction, eg: "data/nirc2_distort_X_post20150413_v1.fits"
    :param distYgeoim: Distortion Y correction, eg:'data/nirc2_distort_Y_post20150413_v1.fits',
    :param src: Source directory to do distortion correction on, eg: 'data/calibrating/darkandflatsubtraction/'
    """

    x_dist = fits.getdata(distXgeoim)
    y_dist = fits.getdata(distYgeoim)

    files = os.listdir(src)

    for i in files:
        data = fits.getdata(src + i)
        header = fits.getheader(src + i)

        new = undistort(data, x_dist, y_dist, src+i)
        fits.writeto('data/calibrating/distort/' + i, new, overwrite=True, header=header)



def createMasterflats():
    """
    Function which creates the master flats
    :return:
    """

    overwrite = False
    flatpath = "data/flats/"

    flatfilelist = ["data/flats/" + x for x in os.listdir("data/flats") if x[-4:] == 'fits']

    finaldir = os.path.join(flatpath, 'final')
    summaryfilename = 'flat_combo_description.txt'

    if overwrite == False and os.path.exists(os.path.join(finaldir, summaryfilename)):
        print('Overwrite = false. Simply returning existing dark summary table.')
        t2 = ascii.read(os.path.join(finaldir, summaryfilename))

    # Create a table of files and with whatever header 20190515 is important
    tab = Table(names=('name', 'framenum', 'itime', 'coadds', 'object', 'filter',
                       'naxis1', 'naxis2', 'multisam', 'sampmode', 'samprate'),
                dtype=(str, np.int16, float, np.int16, str, str,
                       np.int16, np.int16, np.int16, np.int16, np.int16))

    for i in range(len(flatfilelist)):  # Populate the table

        name = flatfilelist[i].split('/')[-1]
        hdulist = fits.open(flatfilelist[i], ignore_missing_end=True)
        a = hdulist[0].header['FRAMENO']
        b = hdulist[0].header['ITIME']
        c = hdulist[0].header['COADDS']
        i = hdulist[0].header['OBJECT']
        j = hdulist[0].header['FILTER']
        d = hdulist[0].header['NAXIS1']
        e = hdulist[0].header['NAXIS2']
        f = hdulist[0].header['MULTISAM']
        g = hdulist[0].header['SAMPMODE']
        h = hdulist[0].header['SAMPRATE']
        tab.add_row((name, a, b, c, i, j, d, e, f, g, h))

    tab.add_index('framenum')

    # make separate summary table of unique flatfield configs
    summarytab = unique(tab, keys=('coadds', 'object', 'itime', 'naxis1', 'naxis2', 'sampmode', 'samprate'))
    summarytab.remove_columns(['name', 'framenum'])
    summarytab.sort(['itime', 'coadds'])
    summarytab['combo'] = range(len(summarytab))
    summarytab = summarytab[['combo', 'object', 'itime', 'coadds', 'naxis1', 'naxis2', 'sampmode', 'samprate']]
    summarytab.add_index('combo')
    # t2

    if not os.path.isdir(finaldir):
        os.mkdir(finaldir)

    summarytab.write(os.path.join(finaldir, summaryfilename),
                     format='ascii.csv', overwrite=True)

    pctcut = 10

    verbose = True
    for i in range(len(summarytab)):
        if verbose:
            print('combo' + str(i))

        # get list of files that match this config
        idxs = [True] * len(tab)
        for key in ['coadds', 'object', 'itime', 'naxis1', 'naxis2', 'sampmode']:
            idxs = idxs & (tab[key] == summarytab[i][key])
        matching_files = list(tab[idxs]['name'].data)

        # read the files
        imgarr = np.zeros([sum(idxs), summarytab[i]['naxis2'], summarytab[i]['naxis1']])
        for ct in range(len(matching_files)):
            with fits.open(os.path.join(flatpath, matching_files[ct]),
                           ignore_missing_end=True) as hdulist:
                imgarr[ct, :, :] = hdulist[0].data

        # Combine frames with top/bottom trimmed mean (better for smaller samples)
        # Keep at single precision
        finalimg = np.single(trim_mean(imgarr, proportiontocut=pctcut / 100, axis=0))

        # make header
        newhdr = fits.Header()

        keys = ['itime', 'object', 'coadds', 'sampmode']
        comments = ['Integration time [sec]', 'Configuration', 'Number of coadds', '1=Single; 2=CDS; 3=MCDS']

        for ct in range(len(keys)):
            newhdr[keys[ct].upper()] = (summarytab[i][keys[ct]], comments[ct])

        for ct in range(len(matching_files)):
            newhdr['IN' + str(ct)] = (matching_files[ct], 'Input file ' + str(ct))

        newhdr['HISTORY'] = 'Combined with trimmed mean with upper/lower %g%% cut' % pctcut

        # make a new FITS file
        hdu = fits.PrimaryHDU(finalimg, newhdr)
        hdul = fits.HDUList([hdu])

        finalname = 'flat_' + finaldir.split('/')[-3] + '_combo' + str(i) + '.fits'
        hdul.writeto(os.path.join(finaldir, finalname), overwrite=True)

def makeMasterDark(csvTable):
    """
    Function to create the master darks
    :param csvTable: The csv table with all the calibration files, ex: "data/KOAlog_cal_n2_23646.csv"
    """
    data = pd.read_csv(csvTable)
    darks = data[data['imagetyp'] == 'dark']
    times = np.unique(darks['elaptime']) # The sets of the darks which will be made

    for i in times:
        specificdarks = darks[darks['elaptime'] == i]
        darkdata = []
        names = specificdarks['koaid'] # List of all the dark files with the given exposure time
        #  print(names)
        for c in names:
            data = fits.getdata('data/darks/' + c)
            darkdata.append(data)

        newmaster = np.mean(darkdata, axis=0)

        fits.writeto('data/darks/masterdark' + str(i) + '.fits', newmaster, overwrite=True)

def img_crop_xy(img, cx, cy, hwx, hwy):
        """
        returns a cropped image, centered on pixel (cx,cy) (0-indexed).
        cropped_img = img_crop_xy(img, cx, cy, hwx, hwy)

        required inputs:
        - img 2D array
        - cx , cy : integer pixel numbers for image center
        - hwx, hwy : half-width of crop in x and y
        eg: hwx=2 creates a 5 pixel-wide image (cx-2, cx-1, cx, cx+1, cx+2)
        """
        if img.ndim > 3 or img.ndim < 2:
            raise Exception('Input can only be a 2D or 3D array')

        # don't modify the original image
        # remember that np passes by object, so just cropping or replacing values
        #   preserves links to original. Change values first to force a copy.
        img = 1.0 * img

        # turn 2D into 3D to keep same syntax
        if img.ndim == 2:
            twoD = True
            img = img[None, :, :]
        else:
            twoD = False

        if hwx > cx:
            raise Exception('hwx must be <=cx')
        if (hwx + cx) > img.shape[2]:
            raise Exception('hwx+dx must be <= img.size[2]')
        if hwy > cy:
            raise Exception('hwy must be <=cy')
        if (hwy + cy) > img.shape[1]:
            raise Exception('hwy+dy must be <= img.size[1]')

        img_crop = img[:, cy - hwy:cy + hwy + 1, cx - hwx:cx + hwx + 1]

        if twoD:
            img_crop = img_crop[0, :, :]

        return img_crop

def rmvinfnan(data):
    """
    Remove the infinites and NaNs from a frame
    :param data: frame
    :return: the frame with no NaNs or infinites
    """
    if np.any(np.isinf(data)) == True:
        temp = np.where(np.isinf(data))
        data[temp] = 0
    if np.any(np.isnan(data)) == True:
        temp = np.where(np.isnan(data))
        data[temp] = 0

    return data

def bpix(coronpath, unsatpath, flatpath):
    """
    Does the bad pixel correction
    :param coronpath: Path for the directory with all the coronagraph science data, eg: 'data/coron_HD134987/coron_star/'
    :param unsatpath: Path for the directory with all the unsaturated data, eg: 'data/coron_HD134987/unsat_star/'
    """
    # In[12]:



    # set path to sci and dark
    scipath = coronpath

    unoccult = False  # to do on unocculted images.
    if unoccult:
        scipath = unsatpath + '0_dark_subtracted/'
        coronpath = unsatpath

    # In[13]:

    scilist = np.sort(glob.glob(scipath + '*.fits'))
    unsatlist = np.sort(glob.glob(unsatpath + '*.fits'))
    flatlist = np.sort(glob.glob(flatpath + '*.fits'))

    for file in scilist:
        print('nan ' + file)
        data, header = fits.getdata(file, header=True)
        if np.any(np.isnan(data)):
            print('Nan detected')
            data = rmvinfnan(data)
            fits.writeto(file, data, header, overwrite=True)

    # In[14]:

    today = date.today()  # grab the date today

    outputpath = coronpath + '2_bad_pixel_corrected'

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)  # make directory to place dark sub files

    radius = 3

    for file in scilist:
        print('main ' + file)
        data, header = fits.getdata(file, header=True)
        im = data

        # This is a trick to quickly compute the mean and standard deviation of pixels in a 2*radius 2*radius box
        # without having to do a for loop over each pixel coordinate
        mean = uniform_filter(im, radius * 2, mode='constant', origin=-radius)
        c2 = uniform_filter(im * im, radius * 2, mode='constant', origin=-radius)
        std = ((c2 - mean * mean) ** .5)

        # identify those more than five sigma away from the mean and replace
        ind = np.where((im > (mean + (5 * std))) | (im < (mean - (5 * std))))
        im[ind] = np.nan
        # if len(bad_ind[0]) > 0:
        #    im[bad_ind] = np.nan
        kernel = Gaussian2DKernel(4.0)
        # note, this replaces all pixels with a nan value… so don’t run this on an image that has been padded with nan values.
        im = interpolate_replace_nans(im, kernel)
        header.append(('COMMENT', '= Bad pixel corrected, Tyler Smith ' + today.strftime("%B %d, %Y")))
        name = file.split('/')[-1]
        name = name.rstrip('.fits') + '_pxlcrctd.fits'
        fits.writeto(os.path.join(outputpath, name), np.float32(im), header, overwrite=True)

    outputpath = unsatlist + '2_bad_pixel_corrected'

    for file in unsatlist:
        data, header = fits.getdata(file, header=True)
        im = data

        # This is a trick to quickly compute the mean and standard deviation of pixels in a 2*radius 2*radius box
        # without having to do a for loop over each pixel coordinate
        mean = uniform_filter(im, radius * 2, mode='constant', origin=-radius)
        c2 = uniform_filter(im * im, radius * 2, mode='constant', origin=-radius)
        std = ((c2 - mean * mean) ** .5)

        # identify those more than five sigma away from the mean and replace
        ind = np.where((im > (mean + (5 * std))) | (im < (mean - (5 * std))))
        im[ind] = np.nan
        # if len(bad_ind[0]) > 0:
        #    im[bad_ind] = np.nan
        kernel = Gaussian2DKernel(4.0)
        # note, this replaces all pixels with a nan value… so don’t run this on an image that has been padded with nan values.
        im = interpolate_replace_nans(im, kernel)
        header.append(('COMMENT', '= Bad pixel corrected, Tyler Smith ' + today.strftime("%B %d, %Y")))
        name = file.split('/')[-1]
        name = name.rstrip('.fits') + '_pxlcrctd.fits'
        fits.writeto(os.path.join(outputpath, name), np.float32(im), header, overwrite=True)

    outputpath = flatpath + '2_bad_pixel_corrected'

    for file in flatlist:
        data, header = fits.getdata(file, header=True)
        im = data

        # This is a trick to quickly compute the mean and standard deviation of pixels in a 2*radius 2*radius box
        # without having to do a for loop over each pixel coordinate
        mean = uniform_filter(im, radius * 2, mode='constant', origin=-radius)
        c2 = uniform_filter(im * im, radius * 2, mode='constant', origin=-radius)
        std = ((c2 - mean * mean) ** .5)

        # identify those more than five sigma away from the mean and replace
        ind = np.where((im > (mean + (5 * std))) | (im < (mean - (5 * std))))
        im[ind] = np.nan
        # if len(bad_ind[0]) > 0:
        #    im[bad_ind] = np.nan
        kernel = Gaussian2DKernel(4.0)
        # note, this replaces all pixels with a nan value… so don’t run this on an image that has been padded with nan values.
        im = interpolate_replace_nans(im, kernel)
        header.append(('COMMENT', '= Bad pixel corrected, Tyler Smith ' + today.strftime("%B %d, %Y")))
        name = file.split('/')[-1]
        name = name.rstrip('.fits') + '_pxlcrctd.fits'
        fits.writeto(os.path.join(outputpath, name), np.float32(im), header, overwrite=True)

#bpix(coronpath='data/coron_HD190360/coron_star/', unsatpath='data/unsat_HD190360/unsat_star/',flatpath = 'data/flats/')

def matchFlats(src):
    """
    Figures out which flat should be used for which image based on where the star is centered compared to the flats.
    Saves which flat is needed in the file's header
    :param src: Directory of the files which you would like to match the flats for, ex:
    :return:
    """
    filelist = os.listdir(src)

    H850t = [849, 390]
    H700t = [699, 390]

    for file in filelist:
        if (file[-4:] == 'fits'):
            data = fits.getdata(src+file)
            header = fits.getheader(src+file)
            meaninsq = np.mean(data[(H850t[1]-5):(H850t[1]+5), (H850t[0]-5):(H850t[0]+5)]) # Takes the average value of a pixel  within a square around 850, 390

            if (meaninsq > 5000): # If the average value is greater than 5k, use the flat centered at an X coordinate of 750
                header['flatReq'] = "850"
            elif (np.mean(data[(H700t[1]-5):(H700t[1]+5), (H700t[0]-5):(H700t[0]+5)]) > 5000): # If the average value is greater than 5k around an X coordinate of 750, use the flat centered at an X coordinate of 750
                header['flatReq'] = "700"
            else:
                header['flatReq'] = "N/A"

            print(header['flatReq'])
            print(meaninsq)

            fits.writeto(src+file, data, header=header, overwrite=True)

#print(matchFlats('data/coron_HD190360/coron_star/'))a

def _img_xcorr_shift(shift, frame1, frame2, mask=None):
    """
    Compute the negative of the xcorr between frame1 and frame2 given shift:

    Args:
        shift: [dx, dy] in pixels to shift frame2
        frame1: refernce frame of shape (Y, X)
        frame2: frame to align of shape (Y, X)
        mask: shape of (Y, X) where =1 for good pixels and =0 for bad pixels

    Return:
        negative of xcorr between frame1 and the shifted frame2
    """
    dx, dy = shift
    oldcenter = [frame2.shape[1], frame2.shape[0]]
    frame2_shifted = klip.align_and_scale(frame2, [oldcenter[0] + dx, oldcenter[1] + dy], oldcenter)
    if mask is not None:
        good = np.where(mask > 0)
        frame1 = frame1[good]
        frame2_shifted = frame2_shifted[good]

    return -np.nansum(frame1 * frame2_shifted)

def basicsubtraction(csvFile, targetName, src):
    """
    :param csvFile: should be filename with all the science frames, eg: 'data/KOAlog_sci_n2_23646.csv'
    :param targetName: name of the target, eg: "HD 190360"
    :param src: source directory for the files, eg: "data/coron_HD190360/coron_star/2_bad_pixel_corrected/"
    """


    data = pd.read_csv(csvFile)
    data = data[data['targname'] == targetName]

    files = os.listdir(src)

    names = [x[0:-14] + ".fits" for x in files]

    for i in names:
        print(i)
        file = data[data['koaid'] == i]
        darkneeded = float(file['elaptime'])

        darkpath = 'data/darks/masterdark' + str(darkneeded) + '.fits' # Getting the flat which corresponds to the exposure time on the science frame
        darkdata = fits.getdata(darkpath)

        if src[-22:] != "2_bad_pixel_corrected/" and src[-21:] != "2_bad_pixel_corrected":
            end = '.fits' # If the source files are not bad pixel corrected, they will end in .fits
        else:
            end = '_pxlcrctd.fits' # If the files are bad pixel corrected, they will have a different ending

        ndata = fits.getdata(src + i[0:-5] + end)
        nheader = fits.getheader(src + i[0:-5] + end)
        if (nheader["FILTER"][0] == "H"): # The following code is just ensuring that the correct flat is selected
            if (nheader['flatReq'] == "700"):
                flat = (fits.getdata('data/flats/final/flat_data_combo2.fits') - fits.getdata(
                    'data/flats/final/flat_data_combo1.fits'))
                flat /= np.median(flat)
                fits.writeto('data/calibrating/darkandflatsubtraction/' + i, (ndata - darkdata) / flat, overwrite=True)
            elif (nheader['flatReq'] == '850'):
                flat = (fits.getdata('data/flats/final/flat_data_combo4.fits') - fits.getdata(
                    'data/flats/final/flat_data_combo3.fits'))
                flat /= np.median(flat)
                fits.writeto('data/calibrating/darkandflatsubtraction/' + i, (ndata - darkdata) / flat, overwrite=True)
            else:
                print('bad')
        else:
            flat = fits.getdata('data/flats/final/flat_data_combo7.fits')
            flat /= np.median(flat)
            fits.writeto('data/calibrating/darkandflatsubtraction/' + i, (ndata - darkdata) / flat, overwrite=True)


def find_best_shift(frame, ref_frame, guess_center=None, inner_mask=5, outer_mask=None):
        """
        Finds the best shift based on xcorr

        Args:
            frame: frame to find offset of. Shape of (Y, X)
            ref_frame: reference frame to align frame to. Shape of (Y, X)

        Return:
            (dx, dy): best shift to shift frame to match reference frame
        """
        if guess_center is None:
            guess_center = np.array([frame.shape[1] / 2., frame.shape[0] / 2.])

        if outer_mask is None:
            outer_mask = frame.shape[0] / 2 - 10

        # make mask
        y, x = np.indices(frame.shape)
        r = np.sqrt((x - guess_center[0]) ** 2 + (y - guess_center[1]) ** 2)
        mask = np.ones(frame.shape)
        bad = np.where((r < inner_mask) | (r > outer_mask))
        mask[bad] = 0

        result = scipy.optimize.minimize(_img_xcorr_shift, (0, 0), args=(ref_frame, frame, mask), method="Nelder-Mead")

        return result.x

def findcenters(src, ref850=None, ref700=None, ref850cen=None, ref700cen=None, overwrite=True):
    """
        Finds the center of the star for each science frame
      :param src: directory with all the files, ex: "data/calibrating/distortH/"
      :param ref850: name of the reference frame for H_850 data, ex: "data/coron_HD190360/coron_star/N2.20190515.48714.fits"
      :param ref700: name of the reference frame for H_700 data, ex: "data/calibrating/distortH/N2.20190515.46468.fits"
      :param ref850cen: center of the star of the reference frame for H_850 data, ex: [847.125, 433.375]
      :param ref700cen: center of the star of the reference frame for H_700 data, ex: [694.375, 434.875]
      :return:
    """
    if (ref850 == None and ref700 == None):
        raise Exception('Must be atleast one reference file')
    if ((ref850cen == None and ref850 != None) or (ref700cen == None and ref700 != None) ):
        raise Exception('Center required for reference frame')

    filelist = [src + x for x in os.listdir(src)]

    filelist = np.array(filelist)
    numeric_values = np.array([int(f[-10:-5]) for f in filelist])
    if ref850:
        refFrame1 = fits.getdata(ref850)
    else:
        refFrame1 = fits.getdata(ref700)

    if ref700:
        refFrame2 = fits.getdata(ref700)
    else:
        refFrame2 = refFrame1

    for i in filelist:
        file = fits.getdata(i)
        header = fits.getheader(i)
        try:
            cen = header['PSFCENTX']
            if (overwrite == False):
                continue
        except:
            pass


        if (header['flatReq'] == "850"):
            if (ref850 == None):
                warnings.warn("warning....... No reference file for H_850 data")
                refCenter = ref700cen
                refFrame = refFrame2
            else:
                refFrame = refFrame1
                refCenter = ref850cen

        elif (header['flatReq'] == "700"):
            if (ref700 == None):
                warnings.warn("warning....... No reference file for H_700 data")
                refCenter = ref850cen
                refFrame = refFrame1
            else:
                refFrame = refFrame2
                refCenter = ref700cen

        shift = np.array(find_best_shift(refFrame, file))

        center = shift+refCenter
        print(i, center)

        hdr = fits.getheader(i)
        hdr['PSFCENTX'] = (center[0], 'Star X numpy coord')
        hdr['PSFCENTY'] = (center[1], 'Star Y numpy coord')

        fits.writeto(i, file, overwrite=True, header=hdr)
        plt.imshow(file)
        plt.scatter(center[0], center[1], s=10, color="red")
        plt.title(i)
        plt.show()

def saveheader(src, copyDir, targDir, backupDir=None):
    """
    Copies headers from one directory to another
    :param src: source directory for everything, ex: 'data/calibrating/distort/'
    :param copyDir: source directory for headers to copy, ex: 'data/calibrating/distort-copy/'
    :param targDir: source directory for target to copy into, ex: 'data/calibrating/distort-copy/'
    :param backupDir: source directory for backup header, ex: 'data/coron_HD190360/coron_star/'
    :return:
    """
    names = os.listdir(src)

    for i in names:
        print(i)
        try:
            distortioncorrectedpath = src+i
            distortioncorrected = fits.getdata(distortioncorrectedpath)

            ndata = fits.getheader(copyDir+i)
            if (ndata['FILTER'][0] == "H"):
                fits.writeto(targDir+i, distortioncorrected, overwrite=True, header=ndata)
        except:
            distortioncorrectedpath = src + i
            distortioncorrected = fits.getdata(distortioncorrectedpath)

            ndata = fits.getheader('data/coron_HD134987/coron_star/' + i)

            fits.writeto(src + i, distortioncorrected, overwrite=True, header=ndata)


def count_nans_in_fits(filename):
    """
    Counts the number of Nans in a file
    :param filename: filename of the file in-question
    :return: Number of NaNs in the file
    """
    # Open the FITS file
    with fits.open(filename) as hdul:
        # Iterate over each HDU (Header/Data Unit) in the FITS file
        nan_count = 0
        for hdu in hdul:
            if isinstance(hdu.data, np.ndarray):  # Check if there is data
                # Count NaNs in the data array
                nan_count += np.isnan(hdu.data).sum()
        return nan_count

def pyklip(targName, pad_size):


    files = ['data/calibrating/distortpads/' + x for x in os.listdir('data/calibrating/distort/')]
    files.sort()
    files = np.array(files)

    dataset = NIRC2.NIRC2Data(filepaths=files, find_star='auto')

    dataset.IWA = 75
    dataset.centers = np.array([dataset.centers[:, 0] + pad_size, dataset.centers[:, 1] + pad_size]).transpose()

    parallelized.klip_dataset(dataset, outputdir="pyklipoutput", fileprefix=targName,
                              annuli=1, subsections=8, movement=0.3, numbasis=[1, 5, 10, 15, 20],
                              mode="ADI", spectrum=None, time_collapse='median')

def pad_fits_file(file_name, output_name, pad_size=550):
    with fits.open(file_name) as hdul:
        original_data = hdul[0].data

        original_shape = original_data.shape

        new_shape = (original_shape[0] + 2 * pad_size, original_shape[1] + 2 * pad_size)

        padded_data = np.full(new_shape, np.nan)

        padded_data[pad_size:pad_size + original_shape[0], pad_size:pad_size + original_shape[1]] = original_data

        hdu = fits.PrimaryHDU(padded_data)
        hdul_new = fits.HDUList([hdu])
        hdul_new.writeto(output_name, overwrite=True)

def nanpad(pad_size=550):

    names = os.listdir('data/calibrating/distort/')

    for i in names:
        pad_fits_file('data/calibrating/distort/' + i, 'data/calibrating/distortpads/' + i, pad_size=pad_size)
        print(i)

        distortioncorrectedpath = 'data/calibrating/distortpads/' + i
        distortioncorrected = fits.getdata(distortioncorrectedpath)

        ndata = fits.getheader('data/calibrating/distort/' + i)

        fits.writeto('data/calibrating/distortpads/' + i, distortioncorrected, overwrite=True, header=ndata)

