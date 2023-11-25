#include <iostream>

#include "opencv2/opencv.hpp"

using namespace cv;

using namespace std;

int main(int argc, const char *argv[])
{

	string haarCascadePath = "haarcascade_frontalface_default.xml";

	VideoCapture video(1);

	if (!video.isOpened())
	{
		cerr << "Error: Could not open camera." << endl;
		return -1;
	}

	Mat img;
	CascadeClassifier faceCascade;
	faceCascade.load(haarCascadePath);

	if (faceCascade.empty())
	{
		cerr << "Error: Could not load face detector." << endl;
		return -1;
	}

	while (true)
	{
		if (!video.read(img))
		{
			cerr << "Error: Could not read frame from camera." << endl;
			break;
		}

		vector<Rect> faces;
		faceCascade.detectMultiScale(img, faces, 1.3, 5);

		for (Rect &face : faces)
			rectangle(img, face, Scalar(50, 50, 255), 3);

		imshow("Image", img);

		if (waitKey(1) == 27) // Break the loop if the user presses the 'Esc' key (ASCII 27)
			break;
	}

	return 0;
}