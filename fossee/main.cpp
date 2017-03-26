/*The c++ libraries have been included below.*/

#include<iostream>
#include<cstdlib>
#include<vector>
#include<sstream>
#include<string>
#include<fstream>
#include<stdio.h>

using namespace std;

/*The required set of opencv libraries have been included below.*/

#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face/facerec.hpp"

/*Using the opencv namespace and face namespace for the facial recognition.*/

using namespace cv;
using namespace cv::face;

/*haarcascade file location for face detection has been defined here*/
String fn_haar = "/home/amb/code/graphics/opencv/fossee/haarcascade_frontalface_alt.xml";

/*haarcascade file location for eye detection has been defined here(for Ellen's eyes)*/
String eyes_cascade_name = "/home/amb/code/graphics/opencv/fossee/haarcascade_eye_tree_eyeglasses.xml";

/*The csv file required to match faces for facial recognition. 
Contains the image locations and class labels for Ellen's and Bradley's face. 
Two known faces have been used as Fisherfaces uses LDA which required to classes of pictures.*/
string fn_csv = "/home/amb/code/graphics/opencv/fossee/face_recog.txt";

/*This haarcascade file location contains the xml file which has been defined for jellyfishes by opencv-haarcascade training.*/
string jl_haar = "/home/amb/code/graphics/opencv/fossee/cascade.xml";


/*Reads the haarcascade files to recognize objects. The first parameter is the file location for the CSV file that needs to be passed, the second parameter and third parameters are for reading the images and their class labels from a line and the last parameter is a semi-colon which is used in the file to divide the image location and its class label.*/
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';')
{
	/*Takes in the file through the file location that has been passed as one of the parameters.*/
        std::ifstream file(filename.c_str(), ifstream::in);

	/*To check if a valid file location has been passed.*/
        if (!file)
        {
                string error_message = "No valid input file was given, please check the given filename.";
                CV_Error(CV_StsBadArg, error_message);
        }
    
	/*To read each line from the CSV file.*/
        string line, path, classlabel;

	/*Read one line and parses it.*/
        while (getline(file, line))
        {
                stringstream liness(line);
                getline(liness, path, separator);
                getline(liness, classlabel);

                if(!path.empty() && !classlabel.empty())
                {
                        images.push_back(imread(path, 0));
                        labels.push_back(atoi(classlabel.c_str()));
                }
        }
}

/*The fucntion that is used for detection(Faces in this case), takes the Photo to detect from as an arguement/parameter.*/
void detectFace( Mat frame)
{
        /*Loads the cascade.*/
        CascadeClassifier face_cascade;
        if( !face_cascade.load( fn_haar ) )
        {
                cout << "Error loading face cascade file\n" << endl;
                return;
        };

	/*This will store the various faces that are going to be detected in the photo.*/
        vector<Rect> faces;
        Mat frame_gray;

	/*To convert the photo to grayscale.*/
        cvtColor( frame, frame_gray, CV_BGR2GRAY );

	/*To equalize the photo's histogram.*/
        equalizeHist( frame_gray, frame_gray );

        /*Detects faces.*/
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	/*For each face, draws a rectangle to denote its detection.*/
        for( size_t i = 0; i < faces.size(); i++ )
        {
		Point pt1,pt2;
	
		pt1.x = faces[i].x;
		pt1.y = faces[i].y;

		pt2.x = faces[i].x + faces[i].width;
		pt2.y = faces[i].y + faces[i].height;
		
		/*Draws a rectangle.*/
  		rectangle(frame, pt1 , pt2, Scalar(255,255,255),2,8,0);	
	}

	/*To display the result after moving it to an appropriate location on the screen.*/
	namedWindow("Face Detection", WINDOW_AUTOSIZE);
	moveWindow("Face Detection", 10,10);
        imshow( "Face Detection", frame );
}

/*To detect the eyes of Ellen. The first parameter is the photo to recognise from and the second and third are for comparison of class labels from CSV file to recognise Ellen's face.*/
void detectEyes( Mat frame, vector<Mat>& images, vector<int>& labels )
{
	/*The first one counts the number of recognitions. As we have two known faces, I had to limit the recognition to Ellen's face and eyes.*/
	int count=0,c=0;
        
	/*Loads the cascades*/
        CascadeClassifier face_cascade;
        CascadeClassifier eyes_cascade;

	/*Checks proper loading of the haarcascade files(Face and Eyes).*/
        if( !face_cascade.load( fn_haar ) )
        {
                cout << "Error loading face cascade file\n" << endl;
        return;
        };

        if( !eyes_cascade.load( eyes_cascade_name ) )
        {
                cout << "Error loading eyes cascade file\n" << endl;
                return;
        };

	/*Reads the CSV file for facial recognition.*/
	read_csv(fn_csv, images, labels);

	int im_width = images[0].cols;
	int im_height = images[0].rows;

	/*To use the FisherFace Recognition Algorithm for facial recognition.*/
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);

	/*All the detected faces are stored here.*/
        vector<Rect> faces;
        Mat frame_gray;

	/*Color to grayscale conversion of the photo.*/
        cvtColor( frame, frame_gray, CV_BGR2GRAY );

	/*To equlise the histogram which is essential for detection.*/
        equalizeHist( frame_gray, frame_gray );

        /*Detects faces*/
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	/*Reads each detected face.*/
	for( size_t i = 0; i < faces.size(); i++ )
        {
		/*To store the region of interest, required for recognition.*/
                Mat faceROI = frame_gray( faces[i] );
		
		/*Stores in a rectangle.*/
		Rect face_i = faces[i];
		Mat face = frame_gray(face_i);

		/*Particular face resized for recognition.*/
		Mat face_resized;
		cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

		/*Predicts recognition.*/
		int prediction = model->predict(face_resized);	

		/*To store the eyes of a face.*/
                std::vector<Rect> eyes;

		/*If prediction is successful that is the face is of a known person i.e Ellen's continue to her eyes. The counter is to prevent the model from continuing to Bradley's eyes.*/
		if(prediction == 0 && count<4)
		{
			count++;
			/*In Ellen's face, to detect eyes.*/
                        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

			/*For every eye.*/
                        for( size_t j = 0; j < eyes.size(); j++ )
                        {
				/*To calculate the BGR values of the centroid of her eyes. Her eyes have been denoted as a circle whose centroid is its center.*/
                                Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
                                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                                circle( frame, center, radius, Scalar( 255, 255, 255 ), 4, 8, 0 );

				Vec3b p = frame.at<Vec3b>(center.x, center.y);

				/*To display the BGR values for each eye.*/
				if(c == 0)
				{		
					cout<<"The (B,G,R) values of the centroid of Ellen's left eye is : "<<"( "<<(int)p[0]<<", "<<(int)p[1]<<", "<<(int)p[2]<<" )"<<endl;
					c++;
				}
				else
				{
					cout<<"The (B,G,R) values of the centroid of Ellen's right eye is : "<<"( "<<(int)p[0]<<", "<<(int)p[1]<<", "<<(int)p[2]<<" )"<<endl;
				}

                      	}
							
        	}
	}

	/*Display the result after moving it to an appropriate location.*/
	namedWindow("Ellen's Eye Detection", WINDOW_AUTOSIZE);
	moveWindow("Ellen's Eye Detection",520,10);
        imshow( "Ellen's Eye Detection", frame );
}

/*The fucntion to detect Jellyfishes in the second image provided. Passes the image as the parameter.*/
void detectJellyfish( Mat frame)
{
        /*Loads the cascade.*/
        CascadeClassifier face_cascade;
        if( !face_cascade.load( jl_haar ) )
        {
                cout << "Error loading face cascade file\n" << endl;
                return;
        };

	/*To stores the jellyfishes.*/
        vector<Rect> faces;
        Mat frame_gray;

	/*To convert to grayscale.*/
        cvtColor( frame, frame_gray, CV_BGR2GRAY );
        equalizeHist( frame_gray, frame_gray );

        /*Detects jellyfishes by using its its specific Haarcascade file.*/
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	/*For each jellyfish detected.*/
        for( size_t i = 0; i < faces.size(); i++ )
        {
		/* The points which help to draw the red cross on the centroid of each detected jellyfish.*/ 
		Point pt1, pt2, pt3, pt4;		

		Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
                int radius = cvRound( (faces[i].width + faces[i].height)*0.25 );
                circle( frame, center, radius, Scalar( 255, 255, 255 ), 2, 8, 0 );

		pt1.x = center.x - faces[i].width*0.1;
		pt1.y = center.y + faces[i].width*0.1;

		pt2.x = center.x + faces[i].width*0.1;
		pt2.y = center.y - faces[i].width*0.1;

		pt3.x = center.x + faces[i].width*0.1;
		pt3.y = center.y + faces[i].width*0.1;

		pt4.x = center.x - faces[i].width*0.1;
		pt4.y = center.y - faces[i].width*0.1;

		/*Draws the cross using two lines.*/
		line(frame, pt1, pt2, Scalar(0,0,255), 2, 8, 0);
		line(frame, pt3, pt4, Scalar(0,0,255), 2, 8, 0);
        }

	/*To display the result after moving it to an appropriate location on the screen.*/
        namedWindow("Jellyfish Detection", WINDOW_AUTOSIZE);
        moveWindow("Jellyfish Detection", 10,10);
        imshow( "Jellyfish Detection", frame );
}

/*The main function.*/
int main(int argc, char** argv)
{
	/*The two given images are stored in to the matrices.*/
	Mat photo1 = imread("/home/amb/code/graphics/opencv/fossee/oscarSelfie.jpg");
	Mat photo2 = imread("/home/amb/code/graphics/opencv/fossee/jellyfish.jpg");

	/*The matrix of images and classlabels used to recognize Ellen which are passed through the function later.*/
	vector<Mat> images;
	vector<int> labels;

	/*Function call to detect the faces in the first image.*/
	detectFace(photo1);

	/*Funciton call to recognise Ellen and point out her eyes.*/
	detectEyes(photo1, images, labels);

	/*Funciton call to detect the jellyfishes.*/
	detectJellyfish(photo2);	

	/*Waits until 0 seconds to exit on pressing any key.*/
	waitKey(0);

	return 0;
}
