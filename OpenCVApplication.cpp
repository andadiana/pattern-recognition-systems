// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <fstream>

using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
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
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
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
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
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

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

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
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
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
		Canny(grayFrame,edges,40,100,3);
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
		if (c == 115){ //'s' pressed - snapp the image to a file
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
	for (int i = 0; i<hist_cols; i++)
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


// PRS Lab 1: 1, 2, 4, 6, 3
std::vector<Point2f> readPoints() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	FILE* f = fopen(fname, "r");
	int n;
	fscanf(f, "%d", &n);
	std::vector<Point2f> points;
	for (int i = 0; i < n; i++) {
		float x, y;
		fscanf(f, "%f%f", &x, &y);
		points.push_back(Point2f(x, y));
	}
	fclose(f);

	return points;
}

Mat drawPoints(std::vector<Point2f> points) {
	Mat img(500, 500, CV_8UC3);
	int n = points.size();
	for (int i = 0; i < n; i++) {
		Point2f point = points.at(i);
		if (point.x >= 0 && point.y >= 0 && point.x < 500 && point.y < 500)
			drawMarker(img, cv::Point(point.x, point.y), cv::Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
		//img.at<Vec3b>(point.y, point.x)[0] = 0; //blue
		//img.at<Vec3b>(point.y, point.x)[1] = 0; //green
		//img.at<Vec3b>(point.y, point.x)[2] = 255; //red

	}
	//imshow("Points", img);
	return img;
}

Mat drawPoints(std::vector<Point> points) {
	Mat img(500, 500, CV_8UC3);
	int n = points.size();
	for (int i = 0; i < n; i++) {
		Point point = points.at(i);
		if (point.x >= 0 && point.y >= 0 && point.x < 500 && point.y < 500)
			drawMarker(img, cv::Point(point.x, point.y), cv::Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
		//img.at<Vec3b>(point.y, point.x)[0] = 0; //blue
		//img.at<Vec3b>(point.y, point.x)[1] = 0; //green
		//img.at<Vec3b>(point.y, point.x)[2] = 255; //red

	}
	//imshow("Points", img);
	return img;
}


Point2f fitLineClosedForm(std::vector<Point2f> points) {
	float sumxy = 0, sumx = 0, sumy = 0, sumxsq = 0;
	int n = points.size();
	for (int i = 0; i < n; i++) {
		Point2f p = points.at(i);
		sumxy += p.x * p.y;
		sumx += p.x;
		sumy += p.y;
		sumxsq += (p.x * p.x);
	}

	float theta1 = (n * sumxy - sumx * sumy) / (n * sumxsq - sumx * sumx);
	float theta0 = (sumy - theta1 * sumx) / n;

	return Point2f(theta0, theta1);
}

Point2f fitLineModel2ClosedForm(std::vector<Point2f> points) {
	float sumxy = 0, sumx = 0, sumy = 0, sumDiffSq = 0;
	int n = points.size();
	for (int i = 0; i < n; i++) {
		Point2f p = points.at(i);
		sumxy += p.x * p.y;
		sumx += p.x;
		sumy += p.y;
		sumDiffSq += (p.y * p.y - p.x * p.x);
	}
	float betaterm1 = 2 * sumxy - 2 * sumx * sumy / n;
	float betaterm2 = sumDiffSq + sumx * sumx / n - sumy * sumy / n;
	float beta = -atan2(betaterm1, betaterm2) / 2;

	float ro = (cos(beta) * sumx + sin(beta) * sumy) / n;

	return Point2f(beta, ro);
}

Mat drawLineTheta(Mat img, float theta0, float theta1) {
	float line1, line2;
	Mat newImg;
	img.copyTo(newImg);
	if (abs(theta1) > 1) {
		cout << "IT'S VERTICAL" << endl;
		line1 = -theta0 / theta1;
		line2 = (500 - theta0) / theta1;
		line(newImg, Point(line1, 0), Point(line2, 500), Scalar(0, 255, 0));
	}
	else {
		line1 = theta0;
		line2 = theta0 + theta1 * 500;
		line(newImg, Point(0, line1), Point(500, line2), Scalar(0, 255, 0));
	}

	return newImg;
}

Mat drawLine(Mat img, float beta, float ro) {
	float line1, line2;
	Mat newImg;
	img.copyTo(newImg);
	int rows = img.rows;
	int cols = img.cols;
	if (abs(beta) < 0.1 || abs(beta) > PI - 0.1) {
		line1 = ro / cos(beta);
		line2 = (ro - cols * sin(beta)) / cos(beta);
		line(newImg, Point(line1, 0), Point(line2, rows), Scalar(0, 255, 0));
	}
	else {
		line1 = ro / sin(beta);
		line2 = (ro - rows * cos(beta)) / sin(beta);
		line(newImg, Point(0, line1), Point(cols, line2), Scalar(0, 255, 0));
	}

	return newImg;
}

void gradientDescent(std::vector<Point2f> points, Mat img) {
	float theta1 = 0.0;
	float theta0 = 0.0;
	float theta0New = 0.0;
	float theta1New = 0.0;
	float alpha = 0.0000009;
	int n = points.size();
	Mat newImg = drawLineTheta(img, theta0, theta1);
	imshow("New img", newImg);
	waitKey();
	float oldCost = 0;
	float newCost = 0;
	do {
		theta0 = theta0New;
		theta1 = theta1New;
		float derivtheta0 = 0.0;
		float derivtheta1 = 0.0;
		oldCost = 0.0;
		for (int i = 0; i < n; i++) {
			Point2f point = points.at(i);
			derivtheta0 += theta0 + theta1 * point.x - point.y;
			derivtheta1 += (theta0 + theta1 * point.x - point.y) * point.x;

			oldCost += pow(theta0 + theta1 * point.x - point.y, 2);
		}
		oldCost = oldCost / 2;

		theta0New = theta0 - alpha * derivtheta0;
		theta1New = theta1 - alpha * derivtheta1;
		Mat newImg = drawLineTheta(img, theta0New, theta1New);
		cout << "Theta 0: " << theta0 << "\n" << "Theta 1: " << theta1 << endl;
		cout << "New Theta 0: " << theta0New << "\n" << "New Theta 1: " << theta1New << endl;

		newCost = 0.0;
		for (int i = 0; i < n; i++) {
			Point2f point = points.at(i);
			newCost += pow(theta0New + theta1New * point.x - point.y, 2);
		}

		imshow("Img", newImg);
		waitKey();
	} while (abs(oldCost - newCost) > 0.001);

}
void leastSquares() {
	std::vector<Point2f> points = readPoints();
	Mat img = drawPoints(points);
	Point2f theta = fitLineClosedForm(points);
	cout << "Theta 1: " << theta.y << "\n" << "Theta 0: " << theta.x << endl;

	Point2f betaro = fitLineModel2ClosedForm(points);
	cout << "Beta: " << betaro.x << "\n" << "Ro: " << betaro.y << endl;

	//img = drawLine(img, betaro.x, betaro.y);
	//img = drawLineTheta(img, theta.x, theta.y);
	//imshow("Points", img);
	gradientDescent(points, img);
	waitKey();
}

vector<Point> readPointsFromImage(Mat img) {

	vector<Point> points;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0) {
				points.push_back(Point(j, i));
			}
		}
	}

	return points;
}

vector<Point> readWhitePointsFromImage(Mat img) {

	vector<Point> points;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 255) {
				points.push_back(Point(j, i));
			}
		}
	}

	return points;
}

float distance(Point p, float a, float b, float c) {
	float dist = fabs(a * p.x + b * p.y + c) / sqrt(a*a + b * b);
	return dist;
}

void ransacLine() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		std::vector<Point> points = readPointsFromImage(img);
		float q = 0.3;
		float t = 10;
		float p = 0.99;
		int bestModel = 0;
		float bestA;
		float bestB;
		float bestC;
		Point bestp1;
		Point bestp2;

		int nr_iterations = log(1 - p) / log(1 - q * q);
		int n = points.size();

		srand(time(NULL));

		for (int i = 0; i < nr_iterations; i++) {
			// Select random point
			int s1, s2;
			s1 = rand() % n;
			s2 = rand() % n;
			while (s1 == s2) {
				s1 = rand() % n;
				s2 = rand() % n;
			}

			// Compute equation of the line
			Point p1 = points.at(s1);
			Point p2 = points.at(s2);
			float a = p1.y - p2.y;
			float b = p2.x - p1.x;
			float c = p1.x * p2.y - p2.x * p1.y;

			int consensusSetCard = 0;
			for (int j = 0; j < n; j++) {
				if (j != s1 && j != s2) {
					if (distance(points.at(j), a, b, c) <= t) {
						consensusSetCard += 1;
					}
				}
			}

			if (consensusSetCard > bestModel) {
				bestA = a;
				bestB = b;
				bestC = c;
				bestModel = consensusSetCard;
				bestp1 = p1;
				bestp2 = p2;

			}
			if (consensusSetCard >= q * n) {
				break;
			}
		}
		Mat newImg;
		img.copyTo(newImg);
		line(newImg, bestp1, bestp2, Scalar(0, 255, 0));

		// draw line
		/*float x1 = -bestC / bestA;
		float x2 = (-bestC - bestB * 500) / bestA;
		line(newImg, Point2f(x1, 0), Point2f(x2, 500), Scalar(0, 255, 0));*/
		imshow("RANSAC", newImg);
		waitKey();
	}
}

int getMaxHough(Mat Hough) {
	int max = 0;
	for (int i = 0; i < Hough.rows; i++) {
		for (int j = 0; j < Hough.cols; j++) {
			if (Hough.at<int>(i, j) > max) {
				max = Hough.at<int>(i, j);
			}
		}
	}
	return max;
}

struct peak {
	float theta;
	int ro;
	int hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

bool isLocalMaximum(Mat Hough, int i, int j, int windowSize) {
	int k = windowSize / 2;
	int max = Hough.at<int>(i, j);
	int maxi = i;
	int maxj = j;
	for (int u = i - k; u <= i + k; u++) {
		for (int v = j - k; v <= j + k; v++) {
			if (u >= 0 && u < Hough.rows && u != i && v != j) {
				int wrapv = (v + 360) % 360;
				if (Hough.at<int>(u, wrapv) > max) {
					max = Hough.at<int>(u, wrapv);
					maxi = u;
					maxj = v;
				}
			}
		}
	}
	if (maxi == i && maxj == j) {
		return true;
	}
	return false;
}

void houghTransformLine() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		std::vector<Point> points = readWhitePointsFromImage(img);
		int dTheta = 1;
		int dRo = 1;
		int k = 10;
		int nrPoints = points.size();
		int diagonal = sqrt(img.cols * img.cols + img.rows * img.rows);

		Mat Hough = Mat::zeros(diagonal + 1, 360, CV_32SC1);

		for (int i = 0; i < nrPoints; i++) {
			for (int theta = 0; theta < 360; theta += dTheta) {
				Point p = points.at(i);
				float thetaRad = theta * PI / 180;
				int ro = p.x * cos(thetaRad) + p.y * sin(thetaRad);

				if (ro >= 0) {
					Hough.at<int>(ro, theta)++;
				}
			}
		}

		// display Hough accumulator
		Mat houghImg;
		int maxHough = getMaxHough(Hough);
		Hough.convertTo(houghImg, CV_8UC1, 255.f / maxHough);

		imshow("Hough accumulator", houghImg);
		
		int windowSize = 7;
		vector<peak> peaks;
		for (int i = 0; i < Hough.rows; i++) {
			for (int j = 0; j < Hough.cols; j++) {
				if (isLocalMaximum(Hough, i, j, windowSize)) {
					float thetaRad = j * PI / 180;
					peaks.push_back(peak{ thetaRad, i, Hough.at<int>(i,j) });
				}
			}
		}

		printf("Nr peaks %d\n", peaks.size());
		Mat imgLines = imread(fname, CV_LOAD_IMAGE_COLOR);
		std::sort(peaks.begin(), peaks.end());

		for (int i = 0; i < k; i++) {
			imgLines = drawLine(imgLines, peaks.at(i).theta, peaks.at(i).ro);
		}

		imshow("Hough Lines", imgLines);
		waitKey(0);

	}
}

Mat distanceTransform(Mat img) {
	Mat dtImg;
	img.copyTo(dtImg);

	Mat_<uchar> c5 = (Mat_<uchar>(3, 3) << 3, 2, 3, 2, 0, 2, 3, 2, 3);
	// first scan: top-down, left-right
	for (int i = 1; i < dtImg.rows; i++) {
		for (int j = 1; j < dtImg.cols - 1; j++) {
			int min_weight = 255;
			for (int k = -1; k <= 0; k++) {
				for (int l = -1; l <= -k; l++) {
					int val = dtImg.at<uchar>(i + k, j + l) + c5.at<uchar>(k + 1, l + 1);
					if (val < min_weight) {
						min_weight = val;
					}
				}
			}
			dtImg.at<uchar>(i, j) = min_weight;
		}
	}

	// second scan: bottom-up, right-left
	for (int i = dtImg.rows - 2; i >= 0; i--) {
		for (int j = dtImg.cols - 2; j >= 1; j--) {
			int min_weight = 255;
			for (int k = 0; k <= 1; k++) {
				for (int l = -k; l <= 1; l++) {
					int val = dtImg.at<uchar>(i + k, j + l) + c5.at<uchar>(k + 1, l + 1);
					if (val < min_weight) {
						min_weight = val;
					}
				}
			}
			dtImg.at<uchar>(i, j) = min_weight;
		}
	}

	return dtImg;
}

float scorePatternMatching(Mat templateImg, Mat unknownDT) {
	float score = 0;
	int nrContour = 0;
	for (int i = 0; i < templateImg.rows; i++) {
		for (int j = 0; j < templateImg.cols; j++) {
			if (templateImg.at<uchar>(i, j) == 0) {
				score += unknownDT.at<uchar>(i, j);
				nrContour++;
			}
		}
	}
	score = score / nrContour;
	return score;
}

void distanceTransformPatternMatching() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat templateImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Template", templateImg);
		
		openFileDlg(fname);
		Mat unknownImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Unknown", unknownImg);

		// Distance transform
		Mat dtImg = distanceTransform(unknownImg);

		// Compute score
		float score = scorePatternMatching(templateImg, dtImg);
		printf("Score is: %f\n", score);
		imshow("DT unknown", dtImg);
		waitKey(0);
	}
}

Mat correlationChart(int f1, int f2, Mat featureMat) {
	Mat chart(256, 256, CV_8UC1);
	for (int i = 0; i < chart.rows; i++) {
		for (int j = 0; j < chart.cols; j++) {
			chart.at<uchar>(i, j) = 255;
		}
	}

	for (int i = 0; i < featureMat.rows; i++) {
		int val1 = featureMat.at<uchar>(i, f1);
		int val2 = featureMat.at<uchar>(i, f2);
		chart.at<uchar>(val1, val2) = 0;
	}

	return chart;
}

void statisticalDataAnalysis() {
	// Load images
	char folder[256] = "Images/Statistical Analysis";
	char fname[256];
	int p = 400;
	int size = 19;
	int N = size*size;
	Mat_<uchar> featureMat = Mat_<uchar>(p, N);

	for (int i = 0; i < p; i++) {
		sprintf(fname, "%s/face%05d.bmp", folder, i + 1);
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		for (int u = 0; u < size; u++) {
			for (int v = 0; v < size; v++) {
				int col = u * size + v;
				uchar val = img.at<uchar>(u, v);
				featureMat.at<uchar>(i, col) = val;
			}
		}
	}
	
	// Compute mean value for all features
	vector<float> means;
	for (int i = 0; i < N; i++) {
		float mean = 0;
		for (int k = 0; k < p; k++) {
			mean += featureMat.at<uchar>(k, i);
		}
		mean = mean / p;
		means.push_back(mean);
	}

	// Compute covariance matrix
	ofstream covarianceFile;
	covarianceFile.open("covariance.csv");
	Mat_<float> covarianceMat = Mat_<float>(N, N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float val = 0;
			for (int k = 0; k < p; k++) {
				val += (featureMat.at<uchar>(k, i) - means.at(i)) * (featureMat.at<uchar>(k, j) - means.at(j));
			}
			val = val / p;
			covarianceMat.at<float>(i, j) = val;
			covarianceFile << val << ",";
		}
		covarianceFile << endl;
	}
	covarianceFile.close();

	// Compute standard deviation for each feature
	vector<float> stdDevs;
	for (int i = 0; i < N; i++) {
		float stdDev = 0;
		for (int k = 0; k < p; k++) {
			stdDev += pow(featureMat.at<uchar>(k, i) - means.at(i), 2);
		}
		stdDev = sqrt(stdDev / p);
		stdDevs.push_back(stdDev);
	}

	// Compute correlation matrix
	ofstream correlationFile;
	correlationFile.open("correlation.csv");
	Mat_<float> correlationMat = Mat_<float>(N, N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float val = covarianceMat.at<float>(i, j) / (stdDevs.at(i) * stdDevs.at(j));
			correlationMat.at<float>(i, j) = val;
			correlationFile << val << ",";
		}
		correlationFile << endl;
	}
	correlationFile.close();
	
	// Correlation charts
	// a) (5, 4), (5, 14)
	int i = 5 * size + 4;
	int j = 5 * size + 14;
	Mat chartA = correlationChart(i, j, featureMat);
	imshow("Chart a", chartA);

	// b) (10, 3), (9, 15)
	i = 10 * size + 3;
	j = 9 * size + 15;
	Mat chartB = correlationChart(i, j, featureMat);
	imshow("Chart b", chartB);

	// c) (5, 4), (18, 0)
	i = 5 * size + 4;
	j = 18 * size;
	Mat chartC = correlationChart(i, j, featureMat);
	imshow("Chart c", chartC);

	waitKey(0);
}

std::vector<Point3i> readPoints(Mat img) {
	std::vector<Point3i> points;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0) {
				points.push_back(Point3i(i, j, -1));
			}
		}
	}
	return points;
}

float pointsDistance(Point3i p, Point3i center) {
	return sqrt(pow(p.x - center.x, 2) + pow(p.y - center.y, 2));
}

void kmeansClustering() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int d = 2;
		vector<Point3i> points = readPoints(img);
		int n = points.size();

		// Initialize the clusters 
		srand(time(NULL));
		int c1, c2, c3;
		c1 = rand() % n;
		c2 = rand() % n;
		while (c1 == c2) {
			c2 = rand() % n;
		}
		c3 = rand() % n;
		while (c1 == c3 || c2 == c3) {
			c3 = rand() % n;
		}

		printf("c1 %d c2 %d c3 %d\n", c1, c2, c3);

		vector<int> randomIndex;
		randomIndex.push_back(c1);
		randomIndex.push_back(c2);
		randomIndex.push_back(c3);

		int K = 3;
		vector<Point3i> centers;

		for (int i = 0; i < randomIndex.size(); i++) {
			points.at(randomIndex.at(i)).z = i;
			centers.push_back(points.at(randomIndex.at(i)));
		}

		// Assignment
		boolean changed = false;
		int cycle = 0;
		do {
			printf("cycle: %d\n", cycle);
			for (int i = 0; i < n; i++) {
				float min;
				if (points.at(i).z == -1) {
					min = img.rows * img.rows + img.cols * img.cols;
				}
				else {
					min = pointsDistance(points.at(i), centers.at(points.at(i).z));
				}
				Point3i kMin;
				changed = false;
				for (int k = 0; k < K; k++) {
					if (points.at(i) != centers.at(k)) {
						float dist = pointsDistance(points.at(i), centers.at(k));
						if (dist < min) {
							changed = true;
							min = dist;
							kMin = centers.at(k);
						}
					}
				}
				if (changed) {
					points.at(i).z = kMin.z;
				}
			}

			// Update centers
			for (int k = 0; k < K; k++) {
				int x = 0;
				int y = 0;
				int nr = 0;
				for (int i = 0; i < n; i++) {
					if (points.at(i).z == k) {
						x += points.at(i).x;
						y += points.at(i).y;
						nr++;
					}
				}
				x = x / nr;
				y = y / nr;
				centers.at(k).x = x;
				centers.at(k).y = y;
			}
			cycle++;
		} while (changed);

		// Assign colors to clusters
		vector<Vec3b> colors;
		for (int i = 0; i < K; i++) {
			//uchar col1 = rand() % 255;
			colors.push_back(Vec3b(rand() % 255, rand() % 255, rand() % 255 ));
		}
		/*colors.push_back({ 255, 0, 0 });
		colors.push_back({ 0, 255, 0 });
		colors.push_back({ 0, 0, 255 });*/

		Mat clustersImg(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

		for (int i = 0; i < points.size(); i++) {
			Point3i p = points.at(i);
			clustersImg.at<Vec3b>(p.x, p.y) = colors.at(p.z);
		}

		imshow("Clusters", clustersImg);

		Mat voronoi(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));
		for (int i = 0; i < voronoi.rows; i++) {
			for (int j = 0; j < voronoi.cols; j++) {
				float min = img.rows * img.rows + img.cols * img.cols;
				Point3i kMin;
				for (int k = 0; k < K; k++) {
					float dist = pointsDistance(Point3i(i, j, -1), centers.at(k));
					if (dist < min) {
						min = dist;
						kMin = centers.at(k);
					}
				}
				voronoi.at<Vec3b>(i, j) = colors.at(kMin.z);
			}
		}
		imshow("Voronoi", voronoi);

		waitKey(0);
	}
}


void principalComponentAnalysis() {
	// Read points from file
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		FILE* f = fopen(fname, "r");
		int n, d;
		fscanf(f, "%d", &n);
		fscanf(f, "%d", &d);
		Mat F(n, d, CV_64FC1);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				float x;
				fscanf(f, "%f", &x);
				F.at<double>(i, j) = x;
			}
		}
		fclose(f);

		// Compute mean for each feature
		vector<float> means;
		for (int k = 0; k < d; k++) {
			float mean = 0;
			for (int i = 0; i < n; i++) {
				mean += F.at<double>(i, k);
			}
			mean = mean / n;
			means.push_back(mean);
		}

		/*for (int i = 0; i < d; i++) {
			printf("Mean %d is: %f\n", i, means.at(i));
		}*/

		// Subtract means
		Mat X(n, d, CV_64FC1);
		for (int i = 0; i < X.rows; i++) {
			for (int j = 0; j < X.cols; j++) {
				X.at<double>(i, j) = F.at<double>(i, j) - means.at(j);
			}
		}

		// Covariance matrix
		Mat C = X.t() * X / (n - 1);

		// Eigenvalue decomposition
		Mat Lambda, Q;
		eigen(C, Lambda, Q);
		Q = Q.t();

		for (int i = 0; i < d; i++) {
			printf("Lambda %d: %f\n", i, Lambda.at<double>(i));
		}

		int k = 3;
		// Extract Qk
		Mat Qk(d, k, CV_64FC1);
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < k; j++) {
				Qk.at<double>(i, j) = Q.at<double>(i, j);
			}
		}

		Mat Xpca;
		Xpca = X * Qk;

		Mat Xkapprox;
		Xkapprox = Xpca * Qk.t();

		// Compute mean absolute difference
		float MAD = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				MAD += abs(X.at<double>(i, j) - Xkapprox.at<double>(i, j));
			}
		}
		MAD = MAD / (n*d);
		printf("Mean absolute difference: %f\n", MAD);

		// Display the image after PCA
		vector<float> mins(k);
		vector<float> maxs(k);
		for (int i = 0; i < k; i++) {
			mins.at(i) = Xpca.at<double>(0, i);
			maxs.at(i) = Xpca.at<double>(0, i);
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++) {
				if (Xpca.at<double>(i, j) < mins.at(j)) {
					mins.at(j) = Xpca.at<double>(i, j);
				}

				if (Xpca.at<double>(i, j) > maxs.at(j)) {
					maxs.at(j) = Xpca.at<double>(i, j);
				}
			}
		}

		Mat img((int)(maxs.at(0) - mins.at(0) + 1),(int)(maxs.at(1) - mins.at(1) + 1), CV_8UC1, Scalar(255));
		for (int i = 0; i < n; i++) {
			if (k == 2) {
				img.at<uchar>((int)(Xpca.at<double>(i, 0) - mins.at(0)), (int)(Xpca.at<double>(i, 1) - mins.at(1))) = 0;
			}
			else if (k == 3) {
				uchar val = (255 / (maxs.at(2) - mins.at(2))) * (Xpca.at<double>(i, 2) - mins.at(2));
				img.at<uchar>((int)(Xpca.at<double>(i, 0) - mins.at(0)), (int)(Xpca.at<double>(i, 1) - mins.at(1))) = 255 - val;
			}
		}

		imshow("After PCA", img);
		waitKey();
	}
}

int* calcHist(Mat img, int nrBins) {
	int histSize = nrBins * 3;
	int* hist = (int*)calloc(histSize, sizeof(int));
	int valsPerBin = 256 / nrBins;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int RVal = img.at<Vec3b>(i, j)[2];
			int GVal = img.at<Vec3b>(i, j)[1];
			int BVal = img.at<Vec3b>(i, j)[0];
			
			hist[(int)(RVal/valsPerBin)]++;
			hist[(int)(nrBins + GVal / valsPerBin)]++;
			hist[(int)(2 * nrBins + BVal / valsPerBin)]++;
		}
	}
	return hist;
}

struct neighbor {
	int index;
	float distance;

	bool operator <(const neighbor &n) const
	{
		return distance < n.distance;
	}
};

int predict(Mat featureVec, Mat labels, Mat img, int histSize, int k, int nrClasses) {
	
	int* hist = calcHist(img, histSize/3);

	// Compute distance
	vector<neighbor> neighbors;
	for (int i = 0; i < featureVec.rows; i++) {
		float dist = 0;
		for (int t = 0; t < histSize; t++) {
			dist += pow(featureVec.at<float>(i, t) - hist[t], 2);
		}
		dist = sqrt(dist);
		neighbors.push_back(neighbor{ i, dist });
	}

	// Sort array and determine k-nearest neighbors
	sort(neighbors.begin(), neighbors.end());

	int* classesVotes = (int*)calloc(nrClasses, sizeof(int));
	for (int i = 0; i < k; i++) {
		classesVotes[labels.at<uchar>(neighbors.at(i).index, 0)]++;
	}

	int maxVotes = 0;
	int maxClass = 0;
	for (int i = 0; i < nrClasses; i++) {
		if (classesVotes[i] > maxVotes) {
			maxVotes = classesVotes[i];
			maxClass = i;
		}
	}
	free(classesVotes);
	
	return maxClass;
}

void knnClassifier() {
	// Read training data
	const int nrClasses = 6;
	char classes[nrClasses][10] = { "beach", "city", "desert", "forest", "landscape", "snow" };

	// Allocate feature matrix and label vector
	int nrInst = 672;

	int rowX = 0;
	int m = 8;
	int k = 7;
	int histSize = 3 * m;
	int featureDim = histSize;
	Mat X(nrInst, featureDim, CV_32FC1);
	Mat y(nrInst, 1, CV_8UC1);

	

	char fname[50];
	//int* hist = (int*)calloc(histSize, sizeof(int));
	for (int c = 0; c < nrClasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "Images/KNN/train/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;

			// Compute histogram
			int* hist = calcHist(img, m);
			for (int d = 0; d < histSize; d++)
				X.at<float>(rowX, d) = hist[d];
			
			y.at<uchar>(rowX) = c;
			rowX++;
		}
	}

	// Confusion matrix
	Mat C(nrClasses, nrClasses, CV_32FC1, Scalar(0));

	// Read test images
	int testInstances = 0;
	for (int c = 0; c < nrClasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "Images/KNN/test/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;
			testInstances++;
			int predictedClass = predict(X, y, img, histSize, k, nrClasses);
			C.at<float>(c, predictedClass)++;
			
		}
	}

	imshow("Confusion matrix", C);
	printf("Confusion matrix:\n");
	for (int i = 0; i < C.rows; i++) {
		for (int j = 0; j < C.cols; j++) {
			printf("%.1f ", C.at<float>(i, j));
		}
		printf("\n");
	}

	// Compute accuracy
	float accuracy = 0;
	for (int i = 0; i < nrClasses; i++) {
		accuracy += C.at<float>(i, i);
	}
	accuracy = accuracy / testInstances;
	printf("Accuracy is: %f\n", accuracy);
	waitKey();
}

void naiveBayes() {
	const int nrClasses = 10;
	char classes[nrClasses][2] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
	int nrinstancesPerClass[nrClasses];
	
	// Allocate feature matrix X and label vector y
	//int instancesPerClass = 100;
	int nrTraining = 60000;
	float cProb = 0.1;
	int nrFeatures = 28 * 28;
	uchar threshold = 128;
	Mat X(nrTraining, nrFeatures, CV_8UC1);
	Mat y(nrTraining, 1, CV_8UC1);
	Mat L255(nrClasses, nrFeatures, CV_32FC1, Scalar(0));
	Mat apriori(nrClasses, 1, CV_32FC1);

	int rowX = 0;
	char fname[50];
	for (int c = 0; c < nrClasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "Images/Naive Bayes/train/%s/%06d.png", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
			//if (img.cols == 0) break;
			if (img.cols == 0) {
				apriori.at<float>(c) = (float)fileNr / nrTraining;
				nrinstancesPerClass[c] = fileNr;
				break;
			}

			// Thresholding
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					if (img.at<uchar>(i, j) < threshold) {
						X.at<uchar>(rowX, i * img.cols + j) = 0;
					}
					else {
						X.at<uchar>(rowX, i * img.cols + j) = 255;
						L255.at<float>(c, i * img.cols + j)++;
					}
				}
			}
			y.at<uchar>(rowX) = c;
			rowX++;
		}
	}

	// Performance of model - compute accuracy
	int totalNrTestInstances = 0;
	int correct = 0;
	for (int cl = 0; cl < nrClasses; cl++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "Images/Naive Bayes/test/%s/%06d.png", classes[cl], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
			//if (img.cols == 0) break;
			if (img.cols == 0) {
				
				break;
			}
			totalNrTestInstances++;

			// Thresholding
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					if (img.at<uchar>(i, j) < threshold) {
						img.at<uchar>(i, j) = 0;
					}
					else {
						img.at<uchar>(i, j) = 255;
					}
				}
			}

			float maxProb = INT_MIN;
			int finalClass = 0;
			for (int c = 0; c < nrClasses; c++) {
				float prob = 0;
				for (int i = 0; i < img.rows; i++) {
					for (int j = 0; j < img.cols; j++) {
						float p = L255.at<float>(c, i * img.cols + j) / nrinstancesPerClass[c];
						if (img.at<uchar>(i, j) == 0) {
							p = 1 - p;
						}
						if (p == 0) {
							p = pow(10, -5);
						}
						prob += log(p);
					}
				}
				prob += log(apriori.at<float>(c));
				if (prob > maxProb) {
					maxProb = prob;
					finalClass = c;
				}
			}
			if (finalClass == cl) {
				correct++;
			}
		}
	}
	float accuracy = (float)correct / totalNrTestInstances;
	printf("Accuracy is %f\n", accuracy);

	char imgname[MAX_PATH];
	while (openFileDlg(imgname)) {
		Mat img = imread(imgname, CV_LOAD_IMAGE_GRAYSCALE);

		// Thresholding
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) < threshold) {
					img.at<uchar>(i, j) = 0;
				}
				else {
					img.at<uchar>(i, j) = 255;
				}
			}
		}

		float maxProb = INT_MIN;
		int finalClass = 0;
		for (int c = 0; c < nrClasses; c++) {
			float prob = 0;
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					float p = L255.at<float>(c, i * img.cols + j) / nrinstancesPerClass[c];
					if (img.at<uchar>(i, j) == 0) {
						p = 1 - p;
					}
					if (p == 0) {
						p = pow(10, -5);
					}
					prob += log(p);
				}
			}
			prob += log(apriori.at<float>(c));
			printf("Prob for class %d is %f\n", c, prob);
			if (prob > maxProb) {
				maxProb = prob;
				finalClass = c;
			}
		}

		printf("Predicted class is: %d\n", finalClass);
		imshow("Image", img);
		waitKey();
	}
}

void perceptronClassifier() {
	// Read points from file
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
		
		int nrFeatures = 3;
		Mat X(0, nrFeatures, CV_32FC1);
		Mat y(0, 1, CV_32FC1);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				Vec3b p = img.at<Vec3b>(i, j);
				float coords[3] = { 1, j, i };
				if (p == Vec3b(255, 0, 0)) {
					// blue point
					X.push_back(Mat(1, 3, CV_32FC1, coords));
					y.push_back(Mat(1, 1, CV_32FC1, Scalar(1)));
				}
				else if (p == Vec3b(0, 0, 255)) {
					// red point
					X.push_back(Mat(1, 3, CV_32FC1, coords));
					y.push_back(Mat(1, 1, CV_32FC1, Scalar(-1)));
				}
			}
		}
		
		Mat W(1, nrFeatures, CV_32FC1, {0.1, 0.1, 0.1});
		int maxIterations = pow(10, 5);
		float Elimit = pow(10, -5);
		int n = X.rows;
		float learningRate = pow(10, -4);
		Mat partialImg;
		img.copyTo(partialImg);
		for (int iter = 0; iter < maxIterations; iter++) {
			float E = 0;
			Mat deriv(1, nrFeatures, CV_32FC1, { 0, 0, 0 });
			for (int i = 0; i < n; i++) {
				float z = 0;
				for (int j = 0; j < W.cols; j++) {
					z += W.at<float>(j) * X.at<float>(i, j);
				}
				if (z * y.at<float>(i) < 0) {
					for (int j = 0; j < X.cols; j++) {
						deriv.at<float>(j) -= y.at<float>(i) * X.at<float>(i, j);
					}
					E++;
				}
			}
			E = E / n;
			for (int j = 0; j < nrFeatures; j++) {
				deriv.at<float>(j) /= n;
			}
			if (E < Elimit)
				break;
			for (int j = 0; j < W.cols; j++) {
				W.at<float>(j) -= learningRate * deriv.at<float>(j);
			}
			/*int y1 = (int)(-W.at<float>(0) / W.at<float>(2));
			int y2 = (int)(-(W.at<float>(0) + img.cols * W.at<float>(1)) / W.at<float>(2));
			printf("y1 is %d y2 is %d\n", y1, y2);
			line(partialImg, Point(0, y1), Point(img.cols, y2), Scalar(0, 255, 0));
			imshow("Perceptron", partialImg);
			waitKey(100);*/
		}
		printf("Weights:\n");
		for (int j = 0; j < W.cols; j++) {
			printf("W[%d]: %f\n", j, W.at<float>(j));
		}

		// Draw line
		Mat newImg;
		img.copyTo(newImg);
		int y1 = (int)(-W.at<float>(0) / W.at<float>(2));
		int y2 = (int)(-(W.at<float>(0) + img.cols * W.at<float>(1)) / W.at<float>(2));
		printf("y1 is %d y2 is %d\n", y1, y2);
		line(newImg, Point(0, y1), Point(img.cols, y2), Scalar(0, 255, 0));
		imshow("Perceptron", newImg);
		waitKey();
	}
}

struct weaklearner {
	int feature_i;
	int threshold;
	int class_label;
	float error;
	int classify(Mat X) {
		if (X.at<float>(feature_i) < threshold)
			return class_label;
		else
			return -class_label;
	}
};

int const MAXT = 100;

struct classifier {
	int T;
	float alphas[MAXT];
	weaklearner hs[MAXT];
	int classify(Mat X) {
		float x = 0;
		for (int t = 0; t < T; t++) {
			x += alphas[t] * hs[t].classify(X);
		}
		if (x < 0) {
			return -1;
		}
		else {
			return 1;
		}
	}
};

weaklearner findWeakLearner(Mat X, Mat y, Mat w, int imgSize) {
	weaklearner bestH = { 0, 0, 0, 0 };
	float bestErr = INT_MAX;
	for (int j = 0; j < X.cols; j++) {
		for (int t = 0; t < imgSize; t++) {
			for (int class_label = -1; class_label < 2; class_label += 2) {
				float err = 0;
				for (int i = 0; i < X.rows; i++) {
					float zi;
					if (X.at<float>(i, j) < t) {
						zi = class_label;
					}
					else {
						zi = -class_label;
					}
					if (zi * y.at<float>(i) < 0) {
						err += w.at<float>(i);
					}
				}
				if (err < bestErr) {
					bestErr = err;
					bestH = { j, t, class_label, err };
				}
			}
		}
	}
	return bestH;
}

void drawBoundary(Mat img, classifier clf) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j) == Vec3b(255, 255, 255)) {
				float v[] = { (float)j ,(float)i };
				Mat X(1, 2, CV_32FC1, v);
				int res = clf.classify(X);
				if (res < 0) {
					img.at<Vec3b>(i, j) = { 255, 153, 153 };
				}
				else {
					img.at<Vec3b>(i, j) = { 153, 255, 255 };
				}
			}
		}
	}
	imshow("Boundary", img);
	waitKey();
}

void adaBoost() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
		int imgSize = img.rows;

		int nrFeatures = 2;
		int nrExamples = 0;
		Mat X(0, nrFeatures, CV_32FC1);
		Mat y(0, 1, CV_32FC1);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				Vec3b p = img.at<Vec3b>(i, j);
				float coords[2] = { j, i };
				if (p == Vec3b(255, 0, 0)) {
					// blue point
					X.push_back(Mat(1, nrFeatures, CV_32FC1, coords));
					y.push_back(Mat(1, 1, CV_32FC1, Scalar(1)));
					nrExamples++;
				}
				else if (p == Vec3b(0, 0, 255)) {
					// red point
					X.push_back(Mat(1, nrFeatures, CV_32FC1, coords));
					y.push_back(Mat(1, 1, CV_32FC1, Scalar(-1)));
					nrExamples++;
				}
			}
		}

		float initialWeight = 1 / (float)nrExamples;
		Mat W(1, nrExamples, CV_32FC1, Scalar(initialWeight));
		classifier c;
		c.T = 10;
		for (int t = 0; t < c.T; t++) {
			weaklearner learner = findWeakLearner(X, y, W, imgSize);
			float alpha = 0.5 * log((1 - learner.error) / learner.error);
			c.alphas[t] = alpha;
			c.hs[t] = learner;
			float s = 0;
			for (int i = 0; i < nrExamples; i++) {
				W.at<float>(i) *= exp(-alpha * y.at<float>(i) * learner.classify(X.row(i)));
				s += W.at<float>(i);
			}
			for (int i = 0; i < nrExamples; i++) {
				W.at<float>(i) /= s;
			}
		}
		
		Mat newImg;
		img.copyTo(newImg);
		drawBoundary(newImg, c);
	}
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
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");

		// Year 4 - PRS Lab1
		printf(" 10 - Least Squares\n");

		// PRS Lab2
		printf(" 11 - RANSAC Line\n");

		// PRS Lab3
		printf(" 12 - Hough Transform for line detection\n");

		// PRS Lab4
		printf(" 13 - Distance transform and pattern matching\n");

		// PRS Lab5
		printf(" 14 - Statistical data analysis\n");

		// PRS Lab6
		printf(" 15 - K-means clustering\n");

		// PRS Lab7
		printf(" 16 - Principal component analysis\n");

		// PRS Lab8
		printf(" 17 - K-nearest neighbors classifier\n");

		// PRS Lab9
		printf(" 18 - Naive Bayes classifier\n");

		// PRS Lab10
		printf(" 19 - Perceptrion classifier\n");

		// PRS Lab11
		printf(" 20 - AdaBoost\n");

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
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			leastSquares();
			break;
		case 11:
			ransacLine();
			break;
		case 12:
			houghTransformLine();
			break;
		case 13:
			distanceTransformPatternMatching();
			break;
		case 14:
			statisticalDataAnalysis();
			break;
		case 15:
			kmeansClustering();
			break;
		case 16:
			principalComponentAnalysis();
			break;
		case 17:
			knnClassifier();
			break;
		case 18:
			naiveBayes();
			break;
		case 19:
			perceptronClassifier();
			break;
		case 20:
			adaBoost();
			break;
		}
	} while (op != 0);
	return 0;
}