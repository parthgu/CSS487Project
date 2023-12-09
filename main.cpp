// 	Author: Dong Nguyen, Parth Gupta, Ka Hin Choi
// 	Purpose: use opencv to detect human face, eyes, smile on video stream
//	and get human emotion happy or sad base on collected faces
//	finally, add filter base on human emotion
// 	Prefix: local computer camera is not available or no face detected
// 	Postfix: add filter tear below human eye when detect emotion sad
//  add rainbow on mouth when detect happy

#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

int main(int argc, const char *argv[])
{
	Mat img;
	Mat imgGray;
	Mat teardrop = imread("teardrop.png", IMREAD_UNCHANGED);
	Mat rainbowsmile = imread("rainbowsmile.png", IMREAD_UNCHANGED);

	string haarCascadePath = "haarcascade_frontalface_default.xml";
	string smileCascadePath = "haarcascade_smile.xml";
	string eyeCascadePath = "haarcascade_eye.xml";

	CascadeClassifier faceCascade;
	CascadeClassifier smileCascade;
	CascadeClassifier eyeCascade;
	// loading cascade
	smileCascade.load(smileCascadePath);
	faceCascade.load(haarCascadePath);
	eyeCascade.load(eyeCascadePath);

	VideoCapture video(1);
	// check if video and cascade works
	if (!video.isOpened())
	{
		cerr << "Error: Could not open camera." << endl;
		return -1;
	}

	if (faceCascade.empty())
	{
		cerr << "Error: Could not load face detector." << endl;
		return -1;
	}

	if (smileCascade.empty())
	{
		cerr << "Error: Could not load smile detector." << endl;
		return -1;
	}

	if (eyeCascade.empty())
	{
		cerr << "Error: Could not load eye detector." << endl;
		return -1;
	}
	// open stream video
	while (true)
	{
		if (!video.read(img))
		{
			cerr << "Error: Could not read frame from camera." << endl;
			video.release();
			break;
		}

		vector<Rect> faces;
		cvtColor(img, imgGray, COLOR_BGR2GRAY);
		faceCascade.detectMultiScale(imgGray, faces, 1.3, 5);
		// detecting faces
		for (Rect &face : faces)
		{
			rectangle(img, face, Scalar(50, 50, 255), 3);
			putText(img, "Face", face.tl() - Point2i(-2, 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(50, 50, 255), 2);

			Mat faceROI = imgGray(face);

			vector<Rect> eyes;
			eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 20, 0 | CASCADE_SCALE_IMAGE, Size(5, 5));

			vector<Rect> smiles;
			smileCascade.detectMultiScale(faceROI, smiles, 1.1, 50, 0 | CASCADE_SCALE_IMAGE, Size(80, 80));
			// detect eyes
			for (Rect &eye : eyes)
			{
				rectangle(img, face.tl() + eye.tl(), face.tl() + eye.br(), Scalar(255, 50, 50), 2);
				putText(img, "Eye", face.tl() + eye.tl() - Point2i(-2, 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 50, 50), 2);

				if (smiles.size() <= 0)
				{
					// Resize teardrop image to fit the size of the detected eye
					Mat resizedTeardrop;
					resize(teardrop, resizedTeardrop, eye.size());

					// Get the region of interest (ROI) for the teardrop
					Mat teardropROI(resizedTeardrop.rows, resizedTeardrop.cols, CV_8UC4, Scalar(0, 0, 0, 0));
					resizedTeardrop(Rect(0, 0, resizedTeardrop.cols, resizedTeardrop.rows)).copyTo(teardropROI);

					// Calculate the position to overlay the teardrop
					Point teardropPosition = face.tl() + eye.tl() + Point(0, eye.height / 1.5);

					// Overlay the teardrop onto the original image
					for (int y = 0; y < teardropROI.rows; ++y)
					{
						for (int x = 0; x < teardropROI.cols; ++x)
						{
							Vec4b teardropPixel = teardropROI.at<Vec4b>(y, x);
							if (teardropPixel[3] > 0) // Check the alpha channel
							{
								img.at<Vec3b>(teardropPosition.y + y, teardropPosition.x + x)[0] = teardropPixel[0];
								img.at<Vec3b>(teardropPosition.y + y, teardropPosition.x + x)[1] = teardropPixel[1];
								img.at<Vec3b>(teardropPosition.y + y, teardropPosition.x + x)[2] = teardropPixel[2];
							}
						}
					}
				}
			}

			for (Rect &smile : smiles)
			{
				rectangle(img, face.tl() + smile.tl(), face.tl() + smile.br(), Scalar(50, 255, 50), 2);
				putText(img, "Smile", face.tl() + smile.tl() - Point2i(-2, 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(50, 255, 50), 2);

				// Resize rainbowsmile image to fit the width of the detected smile
				int resizedRainbowsmileWidth = smile.width;
				int resizedRainbowsmileHeight = static_cast<int>((static_cast<float>(resizedRainbowsmileWidth) / rainbowsmile.cols) * rainbowsmile.rows);

				Mat resizedRainbowsmile;
				resize(rainbowsmile, resizedRainbowsmile, Size(resizedRainbowsmileWidth, resizedRainbowsmileHeight));

				// Calculate the position to overlay the rainbowsmile at the center of the mouth
				Point rainbowsmilePosition = face.tl() + smile.tl() + Point(smile.width / 2 - resizedRainbowsmile.cols / 2, (smile.height / 2 - resizedRainbowsmile.rows / 2) / 1.25);

				// Overlay the rainbowsmile onto the original image with transparency
				for (int y = 0; y < resizedRainbowsmile.rows; ++y)
				{
					for (int x = 0; x < resizedRainbowsmile.cols; ++x)
					{
						Vec4b rainbowsmilePixel = resizedRainbowsmile.at<Vec4b>(y, x);
						if (rainbowsmilePixel[3] > 0) // Check the alpha channel
						{
							Vec3b &imgPixel = img.at<Vec3b>(rainbowsmilePosition.y + y, rainbowsmilePosition.x + x);
							imgPixel[0] = static_cast<uchar>((rainbowsmilePixel[3] / 255.0) * rainbowsmilePixel[0] + (1.0 - rainbowsmilePixel[3] / 255.0) * imgPixel[0]);
							imgPixel[1] = static_cast<uchar>((rainbowsmilePixel[3] / 255.0) * rainbowsmilePixel[1] + (1.0 - rainbowsmilePixel[3] / 255.0) * imgPixel[1]);
							imgPixel[2] = static_cast<uchar>((rainbowsmilePixel[3] / 255.0) * rainbowsmilePixel[2] + (1.0 - rainbowsmilePixel[3] / 255.0) * imgPixel[2]);
						}
					}
				}
			}
		}

		imshow("Image", img);

		if (waitKey(50) == 27) // Break the loop if the user presses the 'Esc' key (ASCII 27)
			break;
	}

	video.release();

	return 0;
}