1. Run crop_faces.py to generate cropped images. These will be printed to new directory: NUAA_cropped
2. Run recognize_alt.py with /NUAA_cropped/.....txt file of cropped images to generate feature extraction and SVM. List of files are:
   1. client_test_raw.txt
   2. imposter_test_raw.txt
   3. client_train_raw.txt
   4. imposter_train_raw.txt
3. metrics.py generates all the metrics
