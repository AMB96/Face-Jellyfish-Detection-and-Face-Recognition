# Face-Jellyfish-Detection-and-Face-Recognition

## How to run ?
Run "cmake ." and then "make" command, the executable is app.

## Details
The repo consists of two images given by the fosseeToolbox to work on. 
Two separate folders of images of Ellen and Bradley to assist with the FisherFace algorithm of facial recognition. 
A CMakeLists.txt file and a main.cpp file are also present, along with them the neccessary Haarcascade files for face, eyes and jellyfish detection are also present. 
Also, a CSV file is present to read Ellen's and Bradley's faces. 
Rest of the files are make files generated after running cmake .

## Dependencies
"cascade.xml" is the haarcascade file for jellyfish that had been generated by haartraining to be able to detect jellyfish. I could only run the haarcascade training for only a few stages due to lack of time, hence certain errors.
"haarcascade_eye_tee_eyeglasses.xml" is the file for detection of eyes. i.e Ellen's
"haarcascade_frontalface_alt.xml" is the file for detection of faces.
"face_recog.txt" is the CSV file to enable facial recognition i.e Ellen's. Edit the image locations inside this file for facial recognition.

Update the file locations of the above files in the "main.cpp" , change to the appropriate location in your system.

## Face detection and Ellen's Eye detection [RESULT]
![2](https://cloud.githubusercontent.com/assets/20037817/24330760/33315332-1243-11e7-9903-dc3d5a2b5422.png)

## Jellyfish Detection [RESULT]
![1](https://cloud.githubusercontent.com/assets/20037817/24332994/0c84cf7c-126e-11e7-96d6-54cd9690e2d9.png)

## For Query
For any query, reach me at "ambwork96@gmail.com"
