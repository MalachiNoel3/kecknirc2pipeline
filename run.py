from main import *

bpix(coronpath='data/coron_HD134987/coron_star/', unsatpath='data/coron_HD134987/unsat_star/',flatpath = 'data/flats/')
print('bpix done')

createMasterflats()
makeMasterDark("data/KOAlog_cal_n2_23646.csv")
matchFlats("data/coron_HD134987/coron_star/2_bad_pixel_corrected/")
print('Darks made and flats done')
basicsubtraction(csvFile='data/KOAlog_sci_n2_23646.csv', targetName="HD 134987", src="data/coron_HD134987/coron_star/2_bad_pixel_corrected/")
print('Subtraction done')
distortioncorrection(distXgeoim='data/nirc2_distort_X_post20150413_v1.fits',
                     distYgeoim='data/nirc2_distort_Y_post20150413_v1.fits',
                     src='data/calibrating/darkandflatsubtraction/')

saveheader(src='data/calibrating/distort/', copyDir='data/calibrating/distort-copy/', targDir='data/calibrating/distort-copy/', backupDir='data/coron_HD190360/coron_star/')
findcenters(src='data/calibrating/distort/', ref700="data/calibrating/distort/N2.20190515.39087.fits", ref850=None, ref700cen=[694.625, 433.625], ref850cen=None)
nanpad(pad_size=550)
pyklip(targName="HD134987", pad_size=550)
