// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <vector>
#include <random>
#include <fstream>
#include "colorConversion.h"
#include "imgproc.hpp"
#include "highgui.hpp"

#define MINLOGRG 0.5
#define MAXLOGRG 2.1
// To segment blue traffic signs
#define MINLOGBG -0.9
#define MAXLOGBG 0.8

/* Definition for ihls segmentation */
// To segment red traffic signs
#define R_HUE_MAX 15 // R_HUE_MAX 11
#define R_HUE_MIN 240
#define R_SAT_MIN 25 // R_SAT_MIN 30
#define R_CONDITION (h < hue_max || h > hue_min) && s > sat_min
// To segment blue traffic signs
#define B_HUE_MAX 163
#define B_HUE_MIN 134
#define B_SAT_MIN 39 // B_SAT_MIN 20
#define B_CONDITION (h < hue_max && h > hue_min) && s > sat_min

using namespace std;
using namespace cv;

int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;

Mat HSeV(Mat bgr)
{

	Mat3b hsv;
	cvtColor(bgr, hsv, COLOR_BGR2HSV);

	Mat1b mask1, mask2;
	inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
	inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);

	Mat1b mask = mask1 | mask2;

	/*String winname = "Test";
	namedWindow(winname);
	moveWindow(winname, 40, 30);*/
	/*imshow(winname, mask);
	waitKey();*/

	return mask;
}
float verifyCircle(cv::Mat dt, cv::Point2f center, float radius, std::vector<cv::Point2f> & inlierSet)
{
	unsigned int counter = 0;
	unsigned int inlier = 0;
	float minInlierDist = 2.0f;
	float maxInlierDistMax = 100.0f;
	float maxInlierDist = radius / 25.0f;
	if (maxInlierDist<minInlierDist) maxInlierDist = minInlierDist;
	if (maxInlierDist>maxInlierDistMax) maxInlierDist = maxInlierDistMax;

	// choose samples along the circle and count inlier percentage
	for (float t = 0; t<2 * 3.14159265359f; t += 0.05f)
	{
		counter++;
		float cX = radius * cos(t) + center.x;
		float cY = radius * sin(t) + center.y;

		if (cX < dt.cols)
			if (cX >= 0)
				if (cY < dt.rows)
					if (cY >= 0)
						if (dt.at<float>(cY, cX) < maxInlierDist)
						{
							inlier++;
							inlierSet.push_back(cv::Point2f(cX, cY));
						}
	}

	return (float)inlier / float(counter);
}


inline void getCircle(cv::Point2f& p1, cv::Point2f& p2, cv::Point2f& p3, cv::Point2f& center, float& radius)
{
	float x1 = p1.x;
	float x2 = p2.x;
	float x3 = p3.x;

	float y1 = p1.y;
	float y2 = p2.y;
	float y3 = p3.y;

	// PLEASE CHECK FOR TYPOS IN THE FORMULA :)
	center.x = (x1*x1 + y1 * y1)*(y2 - y3) + (x2*x2 + y2 * y2)*(y3 - y1) + (x3*x3 + y3 * y3)*(y1 - y2);
	center.x /= (2 * (x1*(y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

	center.y = (x1*x1 + y1 * y1)*(x3 - x2) + (x2*x2 + y2 * y2)*(x1 - x3) + (x3*x3 + y3 * y3)*(x2 - x1);
	center.y /= (2 * (x1*(y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

	radius = sqrt((center.x - x1)*(center.x - x1) + (center.y - y1)*(center.y - y1));
}



std::vector<cv::Point2f> getPointPositions(cv::Mat binaryImage)
{
	std::vector<cv::Point2f> pointPositions;

	for (unsigned int y = 0; y<binaryImage.rows; ++y)
	{
		//unsigned char* rowPtr = binaryImage.ptr<unsigned char>(y);
		for (unsigned int x = 0; x<binaryImage.cols; ++x)
		{
			//if(rowPtr[x] > 0) pointPositions.push_back(cv::Point2i(x,y));
			if (binaryImage.at<unsigned char>(y, x) > 0) pointPositions.push_back(cv::Point2f(x, y));
		}
	}

	return pointPositions;
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}


Mat ransac(Mat color, Mat initial)
{
	//cv::Mat color = cv::imread("../inputData/semi_circle_contrast.png");
	   //cv::Mat color = cv::imread("images/sss.png");
		
	   string ty = type2str(color.type());
	   printf("Matrix: %s\n", ty.c_str());

	    cv::Mat gray,color1;
		color.copyTo(gray);

	    // convert to grayscale
	   // cv::cvtColor(color, gray, CV_BGR2GRAY);
		cv::cvtColor(gray, color1, CV_GRAY2BGR);
		//color.copyTo(gray);

		//color.copyTo(gray);

		//gray = color.clone();
		//Mat gray = color.clone();
	    // now map brightest pixel to 255 and smalles pixel val to 0. this is for easier finding of threshold
	    double min, max;
	    cv::minMaxLoc(gray,&min,&max);
	    float sub = min;
	    float mult = 255.0f/(float)(max-sub);
	    cv::Mat normalized = gray - sub;
	    normalized = mult * normalized;

		/*String winname2 = "Test";
		namedWindow(winname2);

		moveWindow(winname2, 40, 30);*/
		


	   // cv::imshow(winname2, normalized);
	    //--------------------------------
	
	
	    // now compute threshold
	    // TODO: this might ne a tricky task if noise differs...
	    cv::Mat mask;
	    //cv::threshold(input, mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	    cv::threshold(normalized, mask, 100, 255, CV_THRESH_BINARY);
	
	
	
	    std::vector<cv::Point2f> edgePositions;
	    edgePositions = getPointPositions(mask);
	
	    // create distance transform to efficiently evaluate distance to nearest edge
	    cv::Mat dt;
	    cv::distanceTransform(255-mask, dt,CV_DIST_L1, 3);
	
	    //TODO: maybe seed random variable for real random numbers.
	
	    unsigned int nIterations = 0;
	
	    cv::Point2f bestCircleCenter;
	    float bestCircleRadius;
	    float bestCirclePercentage = 0;
	    float minRadius = 50;   // TODO: ADJUST THIS PARAMETER TO YOUR NEEDS, otherwise smaller circles wont be detected or "small noise circles" will have a high percentage of completion
	
	    //float minCirclePercentage = 0.2f;
	    float minCirclePercentage = 0.05f;  // at least 5% of a circle must be present? maybe more...
	
	    int maxNrOfIterations = edgePositions.size();   // TODO: adjust this parameter or include some real ransac criteria with inlier/outlier percentages to decide when to stop
	
	    for(unsigned int its=0; its< maxNrOfIterations; ++its)
	    {
	        //RANSAC: randomly choose 3 point and create a circle:
	        //TODO: choose randomly but more intelligent, 
	        //so that it is more likely to choose three points of a circle. 
	        //For example if there are many small circles, it is unlikely to randomly choose 3 points of the same circle.
	        unsigned int idx1 = rand()%edgePositions.size();
	        unsigned int idx2 = rand()%edgePositions.size();
	        unsigned int idx3 = rand()%edgePositions.size();
	
	        // we need 3 different samples:
	        if(idx1 == idx2) continue;
	        if(idx1 == idx3) continue;
	        if(idx3 == idx2) continue;
	
	        // create circle from 3 points:
	        cv::Point2f center; float radius;
	        getCircle(edgePositions[idx1],edgePositions[idx2],edgePositions[idx3],center,radius);
	
	        // inlier set unused at the moment but could be used to approximate a (more robust) circle from alle inlier
	        std::vector<cv::Point2f> inlierSet;
	
	        //verify or falsify the circle by inlier counting:
	        float cPerc = verifyCircle(dt,center,radius, inlierSet);
	
	        // update best circle information if necessary
	        if(cPerc >= bestCirclePercentage)
	            if(radius >= minRadius)
	        {
	            bestCirclePercentage = cPerc;
	            bestCircleRadius = radius;
	            bestCircleCenter = center;
	        }
	
	    }
	
	    // draw if good circle was found
		cv::Point2f Pos;
		//bestCircleCenter
		Pos.x = bestCircleCenter.x-5;
		Pos.y = bestCircleCenter.y+20;
	    if(bestCirclePercentage >= minCirclePercentage)
	        if(bestCircleRadius >= minRadius);
	
			cv::circle(initial, bestCircleCenter, bestCircleRadius , cv::Scalar(255, 255, 0), 3);
			//cv::circle(original, bestCircleCenter, bestCircleRadius, cv::Scalar(255, 255, 0), 3);
	
			//cv::imshow("outputoriginal", original);
	        
			
			/*String winname = "Test";
			String winname1 = "Test2";
			namedWindow(winname1);
			namedWindow(winname);
			
			moveWindow(winname, 40, 30);
			moveWindow(winname1, 40, 30);*/

			/*cv::imshow(winname1, color1);
			waitKey(0);
	        cv::imshow(winname,mask);
	        cv::waitKey(0);*/
	
	        return initial;
}

void openmultiple()
{
	Mat img = imread("Images/multiple.bmp",
		CV_LOAD_IMAGE_COLOR);

	imshow("res", img);
	waitKey(0);


}
void rgbToHSV() {


	Mat img = imread("Images/Lena_24bits.bmp",
		CV_LOAD_IMAGE_COLOR);

	Mat h(img.rows, img.cols, CV_8UC1);
	Mat s(img.rows, img.cols, CV_8UC1);
	Mat v(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			float r = (float)img.at<Vec3b>(i, j)[2] / 255;
			float g = (float)img.at<Vec3b>(i, j)[1] / 255;
			float b = (float)img.at<Vec3b>(i, j)[0] / 255;

			float M = max(r, max(g, b));
			float m = min(r, min(g, b));
			float C = M - m;

			float V = M;
			float S, H;

			//v.at<uchar>(i, j) = M;

			if (V != 0)
				S = C / V;
			else
				S = 0;

			if (C != 0) {

				if (M == r) H = 60 * (g - b) / C;
				if (M == g) H = 120 + 60 * (b - r) / C;
				if (M == b) H = 240 + 60 * (r - g) / C;

			}
			else
				H = 0;

			if (H < 0)
				H = H + 360;

			H = H * 255 / 360;
			S = S * 255;
			V = V * 255;

			h.at<uchar>(i, j) = H;
			s.at<uchar>(i, j) = S;
			v.at<uchar>(i, j) = V;

		}
	}

	imshow("Hue", h);
	imshow("Saturation", s);
	imshow("Value", v);
	waitKey(0);

}
void storeRedGreenBlue() {

	Mat img = imread("Images/Lena_24bits.bmp",
		CV_LOAD_IMAGE_COLOR);

	Mat b(img.rows, img.cols, CV_8UC1);
	Mat g(img.rows, img.cols, CV_8UC1);
	Mat r(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			b.at<uchar>(i, j) = img.at<Vec3b>(i, j)[0];
			g.at<uchar>(i, j) = img.at<Vec3b>(i, j)[1];
			r.at<uchar>(i, j) = img.at<Vec3b>(i, j)[2];

		}
	}

	imshow("Blue", b);
	imshow("Green", g);
	imshow("Red", r);
	waitKey(0);

}

void grayToBinary()
{
	Mat img = imread("Images/eight.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);


	int threshold;
	scanf("%d", &threshold);



	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) < threshold)
			{
				img.at<uchar>(i, j) = 0;
			}
			else
			{
				img.at<uchar>(i, j) = 255;

			}



		}
	}
	imshow("Binary", img);
	waitKey(0);
}
void testOpenImageLena()
{
	Mat img = imread("Images/Lena_24bits.bmp",
		CV_LOAD_IMAGE_COLOR);

	imshow("Lena", img);
	waitKey(0);
}

void show_sign()
{
	Mat img = imread("Images/sign4.bmp",
		CV_LOAD_IMAGE_COLOR);

	imshow("Lena", img);
	waitKey(0);
}


void convertRGBtoGrayscale()
{
	Mat img = imread("Images/Lena_24bits.bmp",
		CV_LOAD_IMAGE_COLOR);


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);

			int aux = (pixel[0] + pixel[1] + pixel[2]) / 3;

			pixel[0] = aux;
			pixel[1] = aux;
			pixel[2] = aux;

			img.at<Vec3b>(i, j) = pixel;

		}
	}



	imshow("Lena Gray", img);
	waitKey(0);
}
void resize()
{
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);
	int newHeight = 512;
	int newWidth = 512;

	float ratioHeight = (float)img.rows / newHeight;
	float ratioWidth = (float)img.cols / newWidth;

	Mat img2(newHeight, newWidth, CV_8UC1);

	for (int i = 0; i < newHeight - 1; i++) {
		for (int j = 0; j < newWidth - 1; j++) {

			img2.at<uchar>(i, j) = img.at<uchar>(round(i*ratioHeight), round(j*ratioWidth));
		}
	}
	imshow("negative image", img2);
	waitKey(0);
}
void center_crop() {
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);

	Mat img2(256, 256, CV_8UC1);

	for (int i = 64; i < 128 + 64; i++) {
		for (int j = 64; j < 128 + 64; j++) {
			img2.at<uchar>(i, j) = img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img2);
	waitKey(0);
}

void horizontal_flip()
{
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);

	Mat img2(256, 256, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img2.at<uchar>(i, j) = img.at<uchar>(img.rows - 1 - i, j);

		}
	}
	imshow("negative image", img2);
	waitKey(0);

}

void vertical_flip()
{
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);

	Mat img2(256, 256, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img2.at<uchar>(i, j) = img.at<uchar>(i, img.cols - 1 - j);

		}
	}
	imshow("negative image", img2);
	waitKey(0);

}


void create_image()
{
	Mat img(256, 256, CV_8UC3);


	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 128; j++) {
			Vec3b pixel;

			pixel[0] = 255;
			pixel[1] = 255;
			pixel[2] = 255;

			img.at<Vec3b>(i, j) = pixel;

		}
	}

	for (int i = 128; i < 256; i++) {
		for (int j = 128; j < 256; j++) {

			Vec3b pixel;

			pixel[0] = 0;
			pixel[1] = 255;
			pixel[2] = 255;

			img.at<Vec3b>(i, j) = pixel;

		}
	}

	for (int i = 128; i < 256; i++) {
		for (int j = 0; j < 128; j++) {

			Vec3b pixel;

			pixel[0] = 0;
			pixel[1] = 255;
			pixel[2] = 0;
			img.at<Vec3b>(i, j) = pixel;
		}
	}

	for (int i = 0; i < 128; i++) {
		for (int j = 128; j < 256; j++) {

			Vec3b pixel;

			pixel[0] = 0;
			pixel[1] = 0;
			pixel[2] = 255;

			img.at<Vec3b>(i, j) = pixel;

		}
	}



	imshow("negative image", img);
	waitKey(0);
}

void change_grey_additive()
{
	int alpha = 50;
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {


			if (((img.at<uchar>(i, j) + alpha) < 255))
			{
				if (((img.at<uchar>(i, j) + alpha) > 0))
				{
					img.at<uchar>(i, j) = img.at<uchar>(i, j) + alpha;
				}
				else
				{
					img.at<uchar>(i, j) = 0;
				}

			}
			else
			{
				img.at<uchar>(i, j) = 255;
			}


		}
	}


	imshow("negative image", img);
	waitKey(0);
}
void change_grey_multiplicative()
{
	int alpha = 2;
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			//if (((img.at<uchar>(i, j) * alpha) < 255) && (img.at<uchar>(i, j) * alpha) > 0)
			//{

			//	img.at<uchar>(i, j) = img.at<uchar>(i, j) * alpha;

			//}

			//else break;


			if (((img.at<uchar>(i, j) * alpha) < 255))
			{
				img.at<uchar>(i, j) = img.at<uchar>(i, j) * alpha;
			}
			else
			{
				img.at<uchar>(i, j) = 255;
			}

			if (((img.at<uchar>(i, j) * alpha) > 0))
			{
				img.at<uchar>(i, j) = img.at<uchar>(i, j) * alpha;
			}
			else
			{
				img.at<uchar>(i, j) = 0;
			}
		}
	}


	imshow("negative image", img);
	waitKey(0);
}
void negative_image() {
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}

Mat detect_sign() {

	Mat img = imread("Images/sign4.bmp",
		CV_LOAD_IMAGE_COLOR);

	Mat res(img.rows, img.cols, CV_8UC3);

	cv::cvtColor(img, res, CV_BGR2HLS);
	Mat hsv_channels[3];
	cv::split(res, hsv_channels);
	Mat mask;
	cv::inRange(hsv_channels[0], 26, 30, mask);

	
	imshow("Mask", mask);
	waitKey(0);
	return mask;

}
Mat detect_sign_circle(Mat img) {

	//Mat img = imread("Images/signcircle.jpg",
	//	CV_LOAD_IMAGE_COLOR);

	Mat res(img.rows, img.cols, CV_8UC3);

	cv::cvtColor(img, res, CV_BGR2HLS);
	Mat hsv_channels[3];
	cv::split(res, hsv_channels);
	Mat mask;
	cv::inRange(hsv_channels[0], 0, 5, mask);


	String winname = "Test";
	namedWindow(winname);
	moveWindow(winname, 40, 30);

	imshow(winname, mask);
	waitKey(0);
	return mask;

}

void show_sign_circle_original() 
{
	Mat img = imread("Images/signcircle.jpg",
		CV_LOAD_IMAGE_COLOR);

	imshow("Mask", img);
	waitKey(0);

}
void convert_rgb_to_ihls() {


	Mat rgb_image = imread("Images/sign4.bmp",
		CV_LOAD_IMAGE_COLOR);


	// Check the that the image has three channels
	CV_Assert(rgb_image.channels() == 3);

	// Create the output image if needed
	// ihls_image.create(rgb_image.size(), CV_8UC3);
	Mat ihls_image = rgb_image.clone();

	for (auto it = ihls_image.begin<cv::Vec3b>(); it != ihls_image.end<cv::Vec3b>(); ++it) {
		const cv::Vec3b bgr = (*it);
		(*it)[0] = static_cast<uchar> (colorconversion::retrieve_saturation(static_cast<float> (bgr[2]), static_cast<float> (bgr[1]), static_cast<float> (bgr[0])));
		(*it)[1] = static_cast<uchar> (colorconversion::retrieve_luminance(static_cast<float> (bgr[2]), static_cast<float> (bgr[1]), static_cast<float> (bgr[0])));
		(*it)[2] = static_cast<uchar> (colorconversion::retrieve_normalised_hue(static_cast<float> (bgr[2]), static_cast<float> (bgr[1]), static_cast<float> (bgr[0])));
	}


	imshow("Mask", ihls_image);
	waitKey(0);
}

void seg_norm_hue(const cv::Mat& ihls_image, cv::Mat& nhs_image, const int& colour, int hue_max, int hue_min, int sat_min) {

	// Define the different thresholds
	if (colour == 2) {
		if (hue_max > 255 || hue_max < 0 || hue_min > 255 || hue_min < 0 || sat_min > 255 || sat_min < 0) {
			hue_min = R_HUE_MIN;
			hue_max = R_HUE_MAX;
			sat_min = R_SAT_MIN;
		}
	}
	else if (colour == 1) {
		hue_min = B_HUE_MIN;
		hue_max = B_HUE_MAX;
		sat_min = B_SAT_MIN;
	}
	else {
		hue_min = R_HUE_MIN;
		hue_max = R_HUE_MAX;
		sat_min = R_SAT_MIN;
	}

	// Check that the image has three channels
	CV_Assert(ihls_image.channels() == 3);

	// Create the ouput the image
	nhs_image.create(ihls_image.size(), CV_8UC1);

	// I put the if before for loops, to make the process faster.
	// Otherwise for each pixel it had to check this condition.
	// Nicer implementation could be to separate these two for loops in
	// two different functions, one for red and one for blue.
	if (colour == 1) {
		for (int i = 0; i < ihls_image.rows; ++i) {
			const uchar *ihls_data = ihls_image.ptr<uchar>(i);
			uchar *nhs_data = nhs_image.ptr<uchar>(i);
			for (int j = 0; j < ihls_image.cols; ++j) {
				uchar s = *ihls_data++;
				// Although l is not being used and we could have
				// replaced the next line with ihls_data++
				// but for the sake of readability, we left it as it it.
				uchar l = *ihls_data++;
				uchar h = *ihls_data++;
				*nhs_data++ = (B_CONDITION) ? 255 : 0;
			}
		}
	}
	else {
		for (int i = 0; i < ihls_image.rows; ++i) {
			const uchar *ihls_data = ihls_image.ptr<uchar>(i);
			uchar *nhs_data = nhs_image.ptr<uchar>(i);
			for (int j = 0; j < ihls_image.cols; ++j) {
				uchar s = *ihls_data++;
				// Although l is not being used and we could have
				// replaced the next line with ihls_data++
				// but for the sake of readability, we left it as it it.
				uchar l = *ihls_data++;
				uchar h = *ihls_data++;
				*nhs_data++ = (R_CONDITION) ? 255 : 0;
			}
		}
	}

	imshow("Mask", nhs_image);
	waitKey(0);
}

const int alpha_slider_max = 255;
int alpha_slider;
const int beta_slider_max = 255;
int beta_slider;

double alpha;
double beta;

/// Matrices to store images
Mat src;
Mat hsv;
Mat msk;
void on_trackbar1(int, void*)
{
	alpha = (double)alpha_slider / alpha_slider_max;

	cv::cvtColor(src, hsv, CV_BGR2HSV);
	Mat hsv_channels[3];
	cv::split(hsv, hsv_channels);
	cv::inRange(hsv_channels[0], alpha, beta, msk);

	imshow("Segment", msk);
}

void on_trackbar2(int, void*)
{
	beta = (double)beta_slider / beta_slider_max;

	cv::cvtColor(src, hsv, CV_BGR2HSV);
	Mat hsv_channels[3];
	cv::split(hsv, hsv_channels);
	cv::inRange(hsv_channels[0], alpha, beta, msk);

	imshow("Segment", msk);
}

void segment(char *path) {

	/// Read image ( same size, same type )
	src = imread(path);

	/// Initialize values
	alpha_slider = 0;
	beta_slider = 0;

	/// Create Windows
	namedWindow("Segment", 1);

	/// Create Trackbars
	char TrackbarName1[50];
	sprintf(TrackbarName1, "Low x %d", alpha_slider_max);
	char TrackbarName2[50];
	sprintf(TrackbarName2, "High x %d", beta_slider_max);

	createTrackbar(TrackbarName1, "Segment", &alpha_slider, alpha_slider_max, on_trackbar1);
	createTrackbar(TrackbarName2, "Segment", &beta_slider, beta_slider_max, on_trackbar2);

	/// Show some stuff
	on_trackbar1(alpha_slider, 0);
	on_trackbar2(beta_slider, 0);

	/// Wait until user press some key
	waitKey(0);
}
void testOpenImagecameraman() {
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);

	imshow("negative image", img);
	waitKey(0);
}
void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

int area(int r, int g, int b, Mat *src) {

	int area = 0;
	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {

			//Area
			if ((*src).at<Vec3b>(i, j)[2] == r && (*src).at<Vec3b>(i, j)[1] == g && (*src).at<Vec3b>(i, j)[0] == b)
				area++;
		}

	}

	return area;
}

int* center_of_mass(int r, int g, int b, Mat *src) {

	int rmass = 0, cmass = 0;
	int area = 0;
	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {

			if ((*src).at<Vec3b>(i, j)[2] == r && (*src).at<Vec3b>(i, j)[1] == g && (*src).at<Vec3b>(i, j)[0] == b)
				area++;
		}

	}

	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {

			if ((*src).at<Vec3b>(i, j)[2] == r && (*src).at<Vec3b>(i, j)[1] == g && (*src).at<Vec3b>(i, j)[0] == b) {

				rmass += i;
				cmass += j;
			}

		}

	}

	rmass = rmass / area;
	cmass = cmass / area;

	static int center[2];
	center[0] = rmass;
	center[1] = cmass;

	return center;
}

int angle_elongation(int r, int g, int b, Mat *src) {

	double fi;

	double nominator = 0;
	double d1 = 0, d2 = 0;
	double denominator = 0;
	int *center = center_of_mass(r, g, b, src);
	int rmass = *(center);
	int cmass = *(center + 1);


	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {

			if ((*src).at<Vec3b>(i, j)[2] == r && (*src).at<Vec3b>(i, j)[1] == g && (*src).at<Vec3b>(i, j)[0] == b) {

				nominator += (i - rmass) * (j - cmass);
				d1 += (j - cmass) * (j - cmass);
				d2 += (i - rmass) * (i - rmass);
			}

		}

	}

	nominator *= 2;
	denominator = d1 - d2;

	fi = std::atan2(nominator, denominator) / 2;

	double angle = fi * (180 / PI);
	if (angle < 0)
		angle += 180;

	return (int)angle;

}

int perimeter(int r, int g, int b, Mat *src) {

	int perimeter = 0;
	bool has_neighbour = false;


	int test[3] = { -1, 0, 1 };
	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {

			if ((*src).at<Vec3b>(i, j)[2] == r && (*src).at<Vec3b>(i, j)[1] == g && (*src).at<Vec3b>(i, j)[0] == b) {


				for (int x = 0; x < 3; ++x) {

					for (int y = 0; y < 3; ++y) {

						if (test[x] == 0 && test[y] == 0)
							continue;

						if ((*src).at<Vec3b>(i + test[x], j + test[y])[2] != r
							|| (*src).at<Vec3b>(i + test[x], j + test[y])[1] != g
							|| (*src).at<Vec3b>(i + test[x], j + test[y])[0] != b)
						{

							has_neighbour = true;

						}

					}

				}

				if (has_neighbour)
					perimeter++;
				has_neighbour = false;
			}

		}

	}


	return (int)(perimeter * PI / 4);
}

double thin_ratio(int r, int g, int b, Mat *src) {

	int a = area(r, g, b, src);
	int p = perimeter(r, g, b, src);
	return 4 * PI * (double)a / (p * p);

}

double aspect_ratio(int r, int g, int b, Mat *src) {

	int cmin = INT_MAX, cmax = 0, rmin = INT_MAX, rmax = 0;

	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {

			if ((*src).at<Vec3b>(i, j)[2] == r && (*src).at<Vec3b>(i, j)[1] == g && (*src).at<Vec3b>(i, j)[0] == b) {

				if (i < rmin)
					rmin = i;

				if (i > rmax)
					rmax = i;

				if (j < cmin)
					cmin = j;

				if (j > cmax)
					cmax = j;

			}

		}

	}

	return (double)(cmax - cmin + 1) / (rmax - rmin + 1);

}
void draw_test(int r, int g, int b, Mat *src) {

	Mat res((*src).rows, (*src).cols, CV_8UC3);
	int *center = center_of_mass(r, g, b, src);
	Point p(*(center + 1), *center);

	bool has_neighbour = false;
	int test[3] = { -1, 0, 1 };

	for (int i = 0; i < (*src).rows; i++) {
		for (int j = 0; j < (*src).cols; j++) {

			if ((*src).at<Vec3b>(i, j)[2] == r && (*src).at<Vec3b>(i, j)[1] == g && (*src).at<Vec3b>(i, j)[0] == b) {


				for (int x = 0; x < 3; ++x) {

					for (int y = 0; y < 3; ++y) {

						if (test[x] == 0 && test[y] == 0)
							continue;

						if ((*src).at<Vec3b>(i + test[x], j + test[y])[2] != r
							|| (*src).at<Vec3b>(i + test[x], j + test[y])[1] != g
							|| (*src).at<Vec3b>(i + test[x], j + test[y])[0] != b)
						{

							res.at<Vec3b>(i, j)[2] = (*src).at<Vec3b>(i, j)[2];
							res.at<Vec3b>(i, j)[1] = (*src).at<Vec3b>(i, j)[1];
							res.at<Vec3b>(i, j)[0] = (*src).at<Vec3b>(i, j)[0];


						}

					}

				}

			}

		}

	}
	circle(res, p, 1, Scalar(255, 0, 0), 3, 8, 0);
	imshow("Countour", res);
	waitKey(0);
}

void onMouse(int event, int x, int y, int flags, void* param) {

	Mat* src = (Mat*)param;
	int r, g, b;
	if (event == CV_EVENT_LBUTTONDOWN)
	{

		r = (int)(*src).at<Vec3b>(y, x)[2];
		g = (int)(*src).at<Vec3b>(y, x)[1];
		b = (int)(*src).at<Vec3b>(y, x)[0];

		int a = area(r, g, b, src);
		int *center = center_of_mass(r, g, b, src);
		int elongation = angle_elongation(r, g, b, src);
		int p = perimeter(r, g, b, src);
		double thin = thin_ratio(r, g, b, src);
		double aspect = aspect_ratio(r, g, b, src);
		draw_test(r, g, b, src);



		printf("Area: %d\n", a);
		printf("Center of Mass - row %d, column %d\n", *center, *(center + 1));
		printf("Angle of the elongation axis - %d degrees\n", elongation);
		printf("Perimeter with 8 connectivity: %d\n", p);
		printf("Thinness ratio: %.2f \n", thin);
		printf("Aspect ratio: %.2f  \n", aspect);
	}

}
void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);
		setMouseCallback("My Window", onMouse, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}


/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


void colorLabels(int **labels, int rows, int cols) {

	Mat res(rows, cols, CV_8UC3);

	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);

	int max = labels[0][0];

	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			if (max < labels[i][j])
				max = labels[i][j];

	Vec3b *colors = (Vec3b*)malloc(max * sizeof(Vec3b));
	colors[0] = Vec3b(255, 255, 255);
	for (int i = 1; i < max; ++i) {

		uchar b = d(gen);
		uchar g = d(gen);
		uchar r = d(gen);

		colors[i] = Vec3b(b, g, r);
	}

	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j) {

			res.at<Vec3b>(i, j) = colors[labels[i][j]];

		}


	imshow("Colored labels", res);
	waitKey(0);
}

// BFS
int **bfs(char *path) {

	Mat img = imread(path,
		CV_LOAD_IMAGE_GRAYSCALE);

	int label = 0;
	int height = img.rows;
	int width = img.cols;

	std::queue <Point2i> Q;

	int **labels = (int**)malloc(height * sizeof(int*));

	for (int i = 0; i < height; ++i) {
		labels[i] = (int*)malloc(width * sizeof(int));
	}

	int di[8] = { -1, 0, 1, 0, 1, 1, -1, -1 };
	int dj[8] = { 0, -1, 0, 1, 1, -1, 1, -1 };
	uchar neighbors[8];

	//initializae label matrix with 0
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {

			labels[i][j] = 0;

		}
	}

	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j) {

			if (img.at<uchar>(i, j) == 0 && labels[i][j] == 0) {

				label++;

				labels[i][j] = label;
				Q.push({ i, j });

				while (!Q.empty()) {

					Point2i q = Q.front();
					Q.pop();

					//neighbours scan
					for (int k = 0; k < 8; k++) {

						int neighbor_i = q.x + di[k];
						int neighbor_j = q.y + dj[k];
						neighbors[k] = img.at<uchar>(neighbor_i, neighbor_j);


						if (neighbors[k] == 0 && labels[neighbor_i][neighbor_j] == 0) {

							labels[neighbor_i][neighbor_j] = label;
							Q.push({ neighbor_i, neighbor_j });

						}

					}
				}
			}
		}

	return labels;
}

void fromLabelToImageBFS(char *path) {

	Mat img = imread(path,
		CV_LOAD_IMAGE_GRAYSCALE);

	int **labels = bfs(path);
	colorLabels(labels, img.rows, img.cols);
}



//2ways passing

int min_vec(std::vector<int> vec) {

	int min = vec.at(0);
	for (int i = 0; i < vec.size(); ++i)
		if (min > vec.at(i))
			min = vec.at(i);

	return min;
}

int** twoWays(char *path) {

	Mat img = imread(path,
		CV_LOAD_IMAGE_GRAYSCALE);

	int label = 0;
	int height = img.rows;
	int width = img.cols;

	int **labels = (int**)malloc(height * sizeof(int*));

	for (int i = 0; i < height; ++i) {
		labels[i] = (int*)malloc(width * sizeof(int));
	}

	//initializae label matrix with 0
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {

			labels[i][j] = 0;

		}
	}

	int di[4] = { -1, -1, -1,  0, };
	int dj[4] = { -1,  0,  1, -1, };

	std::vector<std::vector<int>> edges;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			if (img.at<uchar>(i, j) == 0 && labels[i][j] == 0) {

				std::vector<int> L;

				for (int k = 0; k < 4; k++) {

					int neighbor_i = i + di[k];
					int neighbor_j = j + dj[k];

					if (labels[neighbor_i][neighbor_j] > 0)
						L.push_back(labels[neighbor_i][neighbor_j]);
				}

				if (L.size() == 0) {

					label++;
					labels[i][j] = label;
					edges.resize(label + 1);

				}
				else {

					int x = min_vec(L);
					labels[i][j] = x;

					for (int y = 0; y < L.size(); ++y)
						if (L.at(y) != x) {

							edges[x].push_back(L.at(y));
							edges[L.at(y)].push_back(x);

						}
				}
			}
		}
	}

	//intermediate

	int newlabel = 0;

	int *newlabels = (int*)malloc((label + 1) * sizeof(int));
	for (int i = 0; i < label + 1; ++i) {
		newlabels[i] = 0;
	}

	for (int i = 1; i <= label; ++i)
		if (newlabels[i] == 0) {

			newlabel++;
			std::queue<int> Q;
			newlabels[i] = newlabel;
			Q.push(i);

			while (!Q.empty()) {

				int x = Q.front();
				Q.pop();

				for (int y = 0; y < edges[x].size(); ++y)
					if (newlabels[edges[x].at(y)] == 0) {

						newlabels[edges[x].at(y)] = newlabel;
						Q.push(edges[x].at(y));
					}

			}

		}

	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			labels[i][j] = newlabels[labels[i][j]];

	return labels;

}

void fromLabelToImageTwoWays(char *path) {

	Mat img = imread(path,
		CV_LOAD_IMAGE_GRAYSCALE);

	int **labels = twoWays(path);
	colorLabels(labels, img.rows, img.cols);
}

struct point {

	int x;
	int y;

};

point getNeighbour(int direction) {

	point result;
	switch (direction) {

	case 0: { result.x = 0; result.y = 1; break; }
	case 1: { result.x = -1; result.y = 1; break; }
	case 2: { result.x = -1; result.y = 0; break; }
	case 3: { result.x = -1; result.y = -1; break; }
	case 4: { result.x = 0; result.y = -1; break; }
	case 5: {result.x = 1; result.y = -1; break; }
	case 6: {result.x = 1; result.y = 0; break; }
	case 7: {result.x = 1; result.y = 1; break; }
	default: {result.x = 0; result.y = 0; break; }
	}
	return result;
}


bool equalPoints(point p1, point p2) {

	if (p1.x == p2.x&&p1.y == p2.y)
		return true;
	return false;
}
ofstream myfile;
ofstream myotherfile;

void borderTracing() {

	Mat src = imread("Images/skew_ellipse.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Initial", src);
	point* borderPoints = (point*)malloc(src.rows*src.cols * sizeof(point));
	int* dir = (int*)malloc(src.rows*src.cols * sizeof(int));

	ofstream myfile;
	ofstream myotherfile;
	myfile.open("codes.txt");
	myotherfile.open("deriv.txt");

	bool ok = true;
	for (int i = 0; i < src.rows&&ok == true; i++)
		for (int j = 0; j < src.cols&&ok == true; j++) {

			if (src.at<uchar>(i, j) == 0) {
				borderPoints[0].x = i;
				borderPoints[0].y = j;
				dir[0] = 7;
				ok = false;
			}
		}

	int i = 0;
	do {
		if (dir[i] % 2 == 0) dir[i + 1] = (dir[i] + 7) % 8;
		else dir[i + 1] = (dir[i] + 6) % 8;
		while (src.at<uchar>(borderPoints[i].x + getNeighbour(dir[i + 1]).x, borderPoints[i].y + getNeighbour(dir[i + 1]).y) != 0)
			dir[i + 1] = (dir[i + 1] + 1) % 8;
		borderPoints[i + 1].x = borderPoints[i].x + getNeighbour(dir[i + 1]).x;
		borderPoints[i + 1].y = borderPoints[i].y + getNeighbour(dir[i + 1]).y;
		myfile << dir[i] << std::endl;
		i++;
	} while (i < 3 || !equalPoints(borderPoints[i - 1], borderPoints[0]) || !equalPoints(borderPoints[i], borderPoints[1]));

	int n = i;

	Mat result = Mat::zeros(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {


			result.at<Vec3b>(i, j)[0] = 255;
			result.at<Vec3b>(i, j)[1] = 255;
			result.at<Vec3b>(i, j)[2] = 255;
		}

	for (int i = 0; i < n; i++) {

		result.at<Vec3b>(borderPoints[i].x, borderPoints[i].y)[2] = 0;
		result.at<Vec3b>(borderPoints[i].x, borderPoints[i].y)[1] = 0;
		result.at<Vec3b>(borderPoints[i].x, borderPoints[i].y)[0] = 0;
	}


	int* deri = (int*)(malloc(n * sizeof(int)));
	for (int i = 0; i < n - 1; i++) {
		deri[i] = dir[i + 1] - dir[i];
		if (deri[i] < 0)
			deri[i] += 8;
		myotherfile << deri[i] << endl;
	}

	imshow("Bordered", result);
	waitKey();

}

void constructFromChainCode(int i, int j) {

	Mat src = imread("Images/oval_vert.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat result = Mat::zeros(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i <= src.rows; i++)
	{
		for (int j = 0; j <= src.cols; j++)
		{
			result.at<uchar>(i, j) = 0;
		}
	}

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {

			result.at<uchar>(i, j) = 255;
		}
	point initialPoint;
	initialPoint.x = i;
	initialPoint.y = j;
	result.at<uchar>(initialPoint.x, initialPoint.y) = 0;
	ifstream f;
	f.open("reconstruct.txt");
	int c;
	while (f >> c) {

		initialPoint.x = initialPoint.x + getNeighbour(c).x;
		initialPoint.y = initialPoint.y + getNeighbour(c).y;
		result.at<uchar>(initialPoint.x, initialPoint.y) = 0;
	}
	imshow("Decoded", result);
	waitKey();
}

Point next(const Point &p, uchar code) {
	int o[8 * 2] = { 0,-1, 1,-1, 1,0, 1,1, 0,1, -1,1, -1,0, -1,-1 };
	return Point(p.x + o[code * 2], p.y + o[code * 2 + 1]);
}
void reconstruct(char *path) {

	int x, y, n;
	Mat src = imread("Images/gray_background.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	std::ifstream file;
	file.open(path);

	file >> x;
	file >> y;
	file >> n;

	Mat res = Mat::zeros(src.rows, src.cols, CV_8UC1);
	int *dir = (int*)malloc(n * sizeof(int));

	for (int i = 0; i < n; ++i) {

		file >> dir[i];

	}

	res.at<uchar>(x, y) = 255;

	for (int i = 0; i < n; ++i) {

		x = x + getNeighbour(dir[i]).x;
		y = y + getNeighbour(dir[i]).y;

		res.at<uchar>(x, y) = 255;

	}

	imshow("Reconstructed", res);
	waitKey(0);
}


Mat dilation(Mat img)
{
	
	Mat copy;
	img.copyTo(copy);

	
		for (int i = 1; i < img.rows - 1; i++)
			for (int j = 1; j < img.cols - 1; j++)
				if (img.at<uchar>(i, j) == 0)
				{
					copy.at< uchar>(i - 1, j) = 0;
					copy.at< uchar>(i - 1, j - 1) = 0;
					copy.at< uchar>(i - 1, j + 1) = 0;
					copy.at< uchar>(i, j - 1) = 0;
					copy.at< uchar>(i, j + 1) = 0;
					copy.at< uchar>(i + 1, j - 1) = 0;
					copy.at< uchar>(i + 1, j + 1) = 0;
					copy.at< uchar>(i + 1, j) = 0;
				}
	
	//imshow("copy_dilation", copy);
	
	
	//waitKey();

	return copy;

}
Mat erosion4x4(Mat img)
{

	Mat copy;
	img.copyTo(copy);


	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++)
			if (img.at<uchar>(i, j) == 0)
			{
				
				copy.at< uchar>(i, j + 1) = 0;
				copy.at< uchar>(i ,j - 1) = 0;
				copy.at< uchar>(i + 1, j) = 0;
				copy.at< uchar>(i - 1, j) = 0;
			}

	//imshow("copy_dilation", copy);


	//waitKey();

	return copy;

}

Mat erosion(Mat img)
{
	Mat copy;
	img.copyTo(copy);

	
		for (int i = 1; i < img.rows - 1; i++)
			for (int j = 1; j < img.cols - 1; j++)
				if (img.at<uchar>(i, j) == 255)
				{
					copy.at< uchar>(i - 1, j) = 255;
					copy.at< uchar>(i - 1, j - 1) = 255;
					copy.at< uchar>(i - 1, j + 1) = 255;
					copy.at< uchar>(i, j - 1) = 255;
					copy.at< uchar>(i, j + 1) = 255;
					copy.at< uchar>(i + 1, j - 1) = 255;
					copy.at< uchar>(i + 1, j + 1) = 255;
					copy.at< uchar>(i + 1, j) = 255;
				}
		
	
	//imshow("copy_erosion", copy);
	//waitKey();
	return copy;
	
	



}

Mat dilation4x4(Mat img)
{
	Mat copy;
	img.copyTo(copy);


	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++)
			if (img.at<uchar>(i, j) == 255)
			{
				
				copy.at< uchar>(i, j - 1) = 255;
				copy.at< uchar>(i, j + 1) = 255;
				copy.at< uchar>(i + 1, j) = 255;
				copy.at< uchar>(i - 1, j) = 255;
			}


	//imshow("copy_erosion", copy);
	//waitKey();
	return copy;





}


Mat open(Mat img) 
{

	Mat erodated = erosion(img);
	Mat dilated = dilation(erodated);

	imshow("opened", dilated);
	waitKey();
	return dilated;
}

Mat close(Mat img)
{
	Mat dilated = dilation(img);
	Mat erodated = erosion(dilated);
	

	imshow("closed imaged", erodated);
	waitKey();
	return erodated;
}

Mat inverse(Mat img) 
{
	Mat copy;
	img.copyTo(copy);

	for (int n = 0; n < 1; n++)
	{
		for (int i = 1; i < img.rows - 1; i++)
			for (int j = 1; j < img.cols - 1; j++)
				if (img.at<uchar>(i, j) == 0)
				{
					copy.at< uchar>(i, j) = 255;
				}
				else 
				{
					copy.at< uchar>(i, j) = 0;
				}
		copy.copyTo(img);
	}
	imshow("inverse", copy);


	waitKey();

	return copy;
}
Mat boundaryExtract(Mat img) 
{
	Mat aux = 255 - erosion(img) + img;
	
	imshow("boundary", aux);


	waitKey();
	return aux;
}

Mat intersection(Mat img1, Mat img2) 
{
	Mat copy;
	for (int i = 1; i < img1.rows - 1; i++)
		for (int j = 1; j < img1.cols - 1; j++)
			if (img1.at<uchar>(i, j) == 0 && img2.at<uchar>(i, j) == 0)
				copy.at<uchar>(i, j) == 0;
	
	return copy;
}
boolean equalMat(Mat a, Mat b)
{
	for (int i = 0; i < (a).rows; i++)
		for (int j = 0; j < (a).cols; j++)
			if ((a).at<uchar>(i, j) != (b).at<uchar>(i, j))
				return false;


	return true;
}
Mat orOperation(Mat a, Mat b)
{
	Mat and (a.rows, a.cols, CV_8UC1);;
	for (int i = 0; i < (a).rows; i++)
		for (int j = 0; j < (a).cols; j++)
			if ((a).at<uchar>(i, j) != (b).at<uchar>(i, j))
				and .at<uchar>(i, j) = 0;
			else
				and.at<uchar>(i, j) = (a).at<uchar>(i, j);

	return and;

}
Mat regionFill(Mat img)
{
	
	Mat copy;
	Mat xk(img.rows, img.cols, CV_8UC1, Scalar(255, 255, 255));
	xk.at<uchar>(img.rows / 2, img.cols / 2) = 0;

	do
	{
		xk.copyTo(copy);
		xk = dilation(copy) + (255 - img);

	} while (equalMat(copy, xk) == false);

	imshow("filling", orOperation(xk, img));
	waitKey();
	return orOperation(xk, img);
}

Mat noise() 
{
	Mat aux = detect_sign();
	Mat aux1;
	Mat aux2;

	//for(int i =0 ; i<3;i++)
	// aux1 = erosion(aux);

	//for (int i = 0; i<3; i++)
		aux2 = dilation(aux);		
		aux1 = erosion(aux2);

		

	imshow("noise", aux1);
	waitKey();
	return aux2;
}
//Mat noise_circle()
//{
//	Mat aux = detect_sign_circle();
//	Mat aux1;
//	Mat aux2;
//
//	//for(int i =0 ; i<3;i++)
//	// aux1 = erosion(aux);
//
//	//for (int i = 0; i<3; i++)
//
//	aux2 = erosion4x4(aux);
//	aux1 = dilation4x4(dilation4x4(dilation4x4(dilation4x4(dilation4x4(dilation4x4(aux2))))));
//	aux2 = aux1;
//	aux1 = erosion4x4(erosion4x4(erosion4x4(erosion4x4(erosion4x4(aux2)))));
//	
//	
//	//extract the noise from the western part of the image
//	for (int i = 0; i < aux1.rows-1; i++)
//		for (int j = 0; j < aux1.cols/2; j++)
//		{
//			if (aux1.at<uchar>(i, j) == 255)
//				aux1.at<uchar>(i,j) = 0;
//		}
//
//	//extract the noise from the bottom-left corner
//	for (int i = aux1.rows/2; i < aux1.rows-1; i++)
//		for (int j = aux1.cols/2; j < aux1.cols -1; j++)
//		{
//			if (aux1.at<uchar>(i, j) == 255)
//				aux1.at<uchar>(i, j) = 0;
//		}
//
//
//	/*aux2 = erosion(erosion(erosion((aux))));
//	aux1 = dilation(dilation(dilation(dilation(aux2))));*/
//
//
//
//	imshow("noise", aux1);
//	waitKey();
//	return aux2;
//}

void drawRandomCircle() 
{

	Mat image = Mat::zeros(400, 400, CV_8UC1);
	for (int i = 0; i < 399; i++) 
	{
		for (int j = 0; j < 399; j++) 
		{
			image.at<uchar>(i, j) = 255;
		}
	}
	// Draw a circle 
	circle(image, Point(200, 200), 32.0, Scalar(0, 0 , 0), 1, 8);
	imshow("Image", image);

	waitKey(0);
	

}

void drawRandomRectangle()
{

	Mat image = imread("Images/sign4.bmp",
		CV_LOAD_IMAGE_COLOR);

	int cercx = 293, cercy = 55;
	int ccercx =260, ccercy = 90;


	circle(image, Point(cercx, cercy), 5.0, Scalar(0, 255, 0), 2, 8);
	circle(image, Point(ccercx, ccercy), 5.0, Scalar(255, 255, 255), 2, 8);
			/*image.at<uchar>(200, 100) = 255;
			image.at<uchar>(230, 130) = 0;*/

		
	

	// Draw a circle 
	//circle(image, Point(200, 200), 32.0, Scalar(0, 0, 0), 1, 8);
	rectangle(image, Point(cercx, cercy), Point(ccercx, ccercy), Scalar(0, 0, 255), 2, 8, 0);
	imshow("Image", image);

	waitKey(0);


}

void histo() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname);
		int height = img.rows;
		int width = img.cols;
		int hist[256];

		for (int i = 0; i < 256; i++)
		{
			hist[i] = 0;
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int g = img.at<uchar>(i, j);
				hist[g]++;
			}
		}
		showHistogram("hist", hist, 256, 200);
		waitKey();
	}

}
void mean_intensity_value() {
	char fname[MAX_PATH];

	if (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int M = src.cols * src.rows;
		float meanValue = 0;
		float standardDeviation = 0;

		int sum = 0;

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				sum += src.at<uchar>(i, j);
			}
		}
		meanValue = sum / M;
		printf("meanValue = %.6f\n", meanValue);
		int secondSum = 0;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				secondSum += (src.at<uchar>(i, j) - meanValue) * (src.at<uchar>(i, j) - meanValue);
			}
		}

		standardDeviation = sqrt(secondSum / M);
		printf("Standard deviation = %.6f\n", standardDeviation);
		waitKey();
	}
}

void compute_histogram() {
	char fname[MAX_PATH];

	if (openFileDlg(fname)) {

		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int hist[256];
		for (int i = 0; i < 256; i++) {
			hist[i] = 0;
		}
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				hist[img.at<uchar>(i, j)] += 1;
			}
		}
		namedWindow("Histogram", 1);
		showHistogram("Histogram", hist, 256, 200);
	}
	waitKey(0);
}

void global_threshold() {
	char fname[MAX_PATH];

	if (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst;
		img.copyTo(dst);
		int hist[256];
		for (int i = 0; i < 256; i++) {
			hist[i] = 0;
		}

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				hist[img.at<uchar>(i, j)] += 1;
			}
		}
		int iMax = 0;
		int iMin = 9999;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) > iMax) iMax = img.at<uchar>(i, j);
				if (img.at<uchar>(i, j) < iMin) iMin = img.at<uchar>(i, j);
			}
		}

		float T = (iMax + iMin) / 2;

		float Tk = T;
		do {
			T = Tk;
			int minN = 0;
			int maxN = 0;
			int tempMean1 = 0;
			int tempMean2 = 0;
			float mean1 = 0;
			float mean2 = 0;
			for (int i = iMin; i <= T; i++) {
				minN += hist[i];
				tempMean1 += i * hist[i];
			}
			for (int i = T + 1; i <= iMax; i++) {
				maxN += hist[i];
				tempMean2 += i * hist[i];
			}

			mean1 = tempMean1 / minN;
			mean2 = tempMean2 / maxN;
			Tk = (mean1 + mean2) / 2;
		} while (abs(Tk - T) < 0.1);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) < T) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = 255;
				}
			}
		}
		printf("Threshold %f \n", Tk);
		imshow("POZA", dst);
		waitKey(0);
	}
}

void histogram_stretching() {

	float gmin, gmax, gres, gamma, offset;
	cout << "gmin: ";
	cin >> gmin;
	cout << endl;

	cout << "gmax: ";
	cin >> gmax;
	cout << endl;

	cout << "gamma: ";
	cin >> gamma;
	cout << endl;

	cout << "offset: ";
	cin >> offset;
	cout << endl;

	char fname[MAX_PATH];

	if (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat img1;
		Mat img2;
		Mat img3;
		img.copyTo(img1);
		img.copyTo(img2);
		img.copyTo(img3);
		int iMax = 0;
		int iMin = 9999;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) > iMax) iMax = img.at<uchar>(i, j);
				if (img.at<uchar>(i, j) < iMin) iMin = img.at<uchar>(i, j);
			}
		}

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {

				img1.at<uchar>(i, j) = gmin + (img.at<uchar>(i, j) - iMin) * (gmax - gmin) / (iMax - iMin);

				img2.at<uchar>(i, j) = 255 * pow(img.at<uchar>(i, j) / 255.0, gamma);

				if (img.at<uchar>(i, j) + offset > 255) img3.at<uchar>(i, j) = 255;
				else if (img.at<uchar>(i, j) + offset < 0) img3.at<uchar>(i, j) = 0;
				else img3.at<uchar>(i, j) = img.at<uchar>(i, j) + offset;
			}
		}
		imshow("Original image", img);
		imshow("Stretching", img1);
		imshow("Gamma Correction", img2);
		imshow("Brightness+", img3);

		waitKey(0);
	}
}

Mat westEmisphere(Mat aux1) 
{
	Mat out;

		for (int i = 0; i < aux1.rows-1; i++)
			for (int j = 0; j < aux1.cols/2; j++)
			{
				if (aux1.at<uchar>(i, j) == 255)
					aux1.at<uchar>(i,j) = 0;
			}
	
		//extract the noise from the bottom-left corner
		for (int i = aux1.rows/2; i < aux1.rows-1; i++)
			for (int j = aux1.cols/2; j < aux1.cols -1; j++)
			{
				if (aux1.at<uchar>(i, j) == 255)
					aux1.at<uchar>(i, j) = 0;
			}

	return aux1;
}
int* get_histogram(Mat src) {

	static int hst[256];

	for (int i = 0; i < 256; ++i)
		hst[i] = 0;

	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {

			hst[src.at<uchar>(i, j)]++;

		}

	return hst;
}

int* get_cumulative_histogram(Mat src) {


	static int *hst = get_histogram(src);

	for (int i = 1; i < 256; ++i) {

		hst[i] = hst[i - 1] + hst[i];

	}

	return hst;
}
void hist_equalization(char *path) {


	Mat src = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	Mat res = Mat::zeros(src.rows, src.cols, CV_8UC1);
	static int* c = get_cumulative_histogram(src);
	static float f[256];

	int M = src.rows * src.cols;

	for (int i = 0; i < 256; ++i) {

		f[i] = (float)c[i] / M;

	}

	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {

			res.at<uchar>(i, j) = 255 * f[src.at<uchar>(i, j)];

		}

	imshow("Equalization", res);
	waitKey(0);

	
}
Mat mean_filter(Mat img)
{
	
	//Mat img = imread("Images/t3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	int size = 5;
	Mat out;

	img.copyTo(out);
	vector<uchar> vector(size*size);

	for (int i = size / 2; i < img.rows - size / 2; i++)
		for (int j = size / 2; j < img.cols - size / 2; j++)
		{
			int counter = 0;
			for (int k = -size / 2; k <= size / 2; k++)
				for (int l = -size / 2; l <= size / 2; l++)
				{

					vector[counter] = img.at<uchar>(i + k, j + l);
					counter++;
				}
			std::sort(vector.begin(), vector.end());
			out.at<uchar>(i, j) = vector[size*size / 2];


		}
	/*String winname = "Test";
	namedWindow(winname);
	moveWindow(winname, 40, 30);*/

	/*imshow(winname, img);*/
	

	/*String winname5 = "out";
	namedWindow(winname5);
	moveWindow(winname5, 40, 30);*/
	/*imshow(winname5, out);
	waitKey(0);*/
	return out;
}
Mat erod(Mat img) {
	return erosion4x4(erosion4x4(erosion4x4(erosion4x4(erosion4x4(img)))));
}
void project() 
{
	cv::Mat color = cv::imread("images/test10.jpg");
	cv::Mat aux = HSeV(color);
	cv::Mat filtered = mean_filter(aux);
	cv::Mat western = westEmisphere(filtered);
	cv::Mat eroded = erod(western);

	string ty = type2str(eroded.type());
	printf("Matrix: %s\n", ty.c_str());

	cv::Mat ransacApplied = ransac(eroded,color);
	String winname = "Final";
	namedWindow(winname);
	moveWindow(winname, 40, 30);
	imshow(winname, ransacApplied);
	waitKey(0);
}



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Resize image\n");
		printf(" 4 - Process video\n");
		printf(" 5 - Snap frame from live video\n");
		printf(" 6 - Mouse callback demo\n");
		printf(" 7 - Negative image for cameraman.bmp\n");
		printf(" 8 - Original image for cameraman\n");
		printf(" 9 - Change grey additive\n");
		printf(" 10 - Change grey multiplicative\n");
		printf(" 11 - Create an image\n");
		printf(" 12 - Vertical flip\n");
		printf(" 13 - Horizontal flip\n");
		printf(" 14 - Center Crop\n");
		printf(" 15 - Resize\n");
		printf(" 16 - Show Lena Original\n");
		printf(" 17 - Convert RGB to Grayscale\n");
		printf(" 18 - Convert Grayscale to Binary\n");
		printf(" 19 - Store Red Green Blue\n");
		printf(" 20 - RGB to HSV\n");
		printf(" 21 - ShowOriginal Sign\n");
		printf(" 22 - Detect Sign\n");
		printf(" 23 - Open multiple\n");
		printf(" 24 - Segment with trackbar\n");
		printf(" 25 - BFS\n");
		printf(" 26 - 2 ways passing\n");
		printf(" 27 - Border\n");
		printf(" 28 - Reconstruct\n");
		printf(" 29 - From RGB to IHLS\n");
		printf(" 30 - Dilation\n");
		printf(" 31 - Erotion\n");
		printf(" 32 - Open\n");
		printf(" 33 - Close\n");
		printf(" 34 - Boundary extraction\n");
		printf(" 35 - Region filling\n");
		printf(" 36 - Inverse\n");
		printf(" 37 - Add 2 images\n");
		printf(" 38 - Noise traffic sign\n");
		printf(" 39 - Draw random circle\n");
		printf(" 40 - Draw random rectangle\n");
		printf(" 41 - Histograma\n");
		printf(" 42 - Mean and variance\n");
		printf(" 43- Compute histogram\n");
		printf(" 44- Global threshold\n");
		printf(" 45- Histogram stretching\n");
		printf(" 46- Histogram equalization\n");
		printf(" 47- Show sign circle original\n");
		printf(" 48- Detect sign circle\n");
		printf(" 49- Noise reduction for circle traffic sign\n");
		printf(" 50- Ransac\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testResize();
			break;
		case 4:
			testVideoSequence();
			break;
		case 5:
			testSnap();
			break;
		case 6:
			testMouseClick();
			break;
		case 7:
			negative_image();
			break;
		case 8:
			testOpenImagecameraman();
			break;
		case 9:
			change_grey_additive();
			break;

		case 10:
			change_grey_multiplicative();
			break;

		case 11:
			create_image();
			break;

		case 12:
			horizontal_flip();
			break;
		case 13:
			vertical_flip();
			break;
		case 14:
			center_crop();
			break;
		case 15:
			resize();
			break;
		case 16:
			testOpenImageLena();
			break;

		case 17:
			convertRGBtoGrayscale();
			break;

		case 18:
			grayToBinary();
			break;

		case 19:
			storeRedGreenBlue();
			break;

		case 20:
			rgbToHSV();
			break;

		case 21:
			show_sign();
			break;

		case 22:
			detect_sign();
			break;

		case 23:
			openmultiple();
			break;

		case 24:
			segment("Images/sign.png");
			break;

		case 25:
			fromLabelToImageBFS("Images/labeling1.bmp");
			fromLabelToImageBFS("Images/labeling2.bmp");

			break;

		case 26:
			fromLabelToImageTwoWays("Images/labeling1.bmp");
			fromLabelToImageTwoWays("Images/labeling2.bmp");

			break;

		case 27:
			borderTracing();

			break;

		case 28:
			//constructFromChainCode(159, 175);
			reconstruct("reconstruct.txt");
			break;

		case 29:
			convert_rgb_to_ihls();
			break;

		case 30:
			dilation(imread("Images/morpho/dilate/mon1thr1_bw.bmp",CV_8UC1));
			break;

		case 31:
			erosion(imread("Images/morpho/dilate/mon1thr1_bw.bmp", CV_8UC1));
			break;

		case 32:
			open(imread("Images/morpho/open/cel4thr3_bw.bmp", CV_8UC1));
			break;

		case 33:
			close(imread("Images/morpho/close/phn1thr1_bw.bmp", CV_8UC1));
			break;

		case 35:
			regionFill(imread("Images/morpho/regfill/reg1neg1_bw.bmp", CV_8UC1));
			break;

		case 34:
			boundaryExtract(imread("Images/morpho/boundext/reg1neg1_bw.bmp", CV_8UC1));
			break;

		case 36:
			inverse(imread("Images/horizontal_ellipse.bmp", CV_8UC1));
			break;

		case 37:
			imshow("",imread("Images/horizontal_ellipse.bmp", CV_8UC1) +imread("Images/horizontal_ellipse.bmp", CV_8UC1));
			waitKey();
			break;

		case 38:
			noise();
			break;
			
		case 39:
			drawRandomCircle();
			break;
		case 40:
			drawRandomRectangle();
			break;

		case 41:
			histo();
			break;
		case 42:
			mean_intensity_value();
			break;
		case 43:
			compute_histogram();
			break;
		case 44:
			global_threshold();
			break;
		case 45:
			histogram_stretching();
			break;
		case 46:
			hist_equalization("Images/hist/Hawkes_Bay_NZ.bmp");
			imshow("Original image", imread("Images/hist/Hawkes_Bay_NZ.bmp", CV_LOAD_IMAGE_GRAYSCALE));
			waitKey();
			break;
		
		case 47:
			show_sign_circle_original();
			break;

		case 48:
			//detect_sign_circle();
			break;
		case 49:
			//noise_circle();
			break;
		case 50:
			project();
			break;

		case 51:
			mean_filter(imread("Images/hsev1.png", CV_LOAD_IMAGE_GRAYSCALE));
			break;

		case 52:
			//ransac();
			break;
		}
	} while (op != 0);
	return 0;
}