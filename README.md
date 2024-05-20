# kecknirc2pipeline

This pipeline can be used to process NIRC2 Keck data in order to find background stars

# Organizing the data

Place all the data in a directory named "data"

In the "data" directory, create the following folders and directories: "calibrating," "coron_{targname}", "darks," "flats," and "unsat_{targname}." Create a "2_bad_pixel_corrected" in the "coron_{targname}" directory. 

Place the science frames with the coronagraph deployed, darks, flats, and unsaturated images in the "coron_{targname}", "darks," "flats," and "unsat_{targname}" folders respectively.

Additionally, place the distortion correction files and the science data and calibration file csvs in the "data" directory. 

# Running the pipeline

Change the folder and target names in the run.py file to match the dataset which you are processing and then run the run.py file. Be careful to pay attention to the spacing. 
