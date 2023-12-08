#include <iostream>
#include "MostVotedBin.cpp"
#include "opencv2/opencv.hpp"

using namespace cv;

using namespace std;

string detectEmotion(const Mat& image) {
	// Read an image
	//cv::Mat image = cv::imread("angry5.jpg");

	// Convert the image to grayscale
	/*Mat imgGray;
	cvtColor(image, imgGray, COLOR_BGR2GRAY);
	*/
	// Load the Haar Cascade classifier for face detection
	std::string haarCascadePath = "haarcascade_frontalface_default.xml";
	cv::CascadeClassifier faceCascade;
	faceCascade.load(haarCascadePath);

	// Detect faces in the image
	vector<cv::Rect> faces;
	faceCascade.detectMultiScale(image, faces, 1.3, 5);

	// Check if at least one face is detected
	if (!faces.empty()) {
		// Assume the first detected face as the region of interest (ROI)
		Rect faceRect = faces[0];

		// Create a new image specifically for the detected face
		Mat faceImage = image(faceRect).clone();  // Use clone to get a deep copy

		int cropHeight = static_cast<int>(faceImage.rows * 0.5);
		int cropWidth = static_cast<int>(faceImage.cols * 0.5);

		// Define start and end positions for cropping
		int x1 = faceImage.cols/4;
		int y1 = faceImage.rows - cropHeight + cropHeight / 2-10;
		int x2 = faceImage.cols*3/4;
		int y2 = faceImage.rows- cropHeight / 4;
		cv::Rect roiRect(x1, y1, x2, y2 - y1);


		// Crop the adjusted face region from the face image
		cv::Mat croppedBottomHalfFace = faceImage(roiRect).clone();  // Use clone to get a deep copy
		////////////////////////////////RAY);
		cv::GaussianBlur(croppedBottomHalfFace, croppedBottomHalfFace, cv::Size(5, 5), 0);
		// Display the original image and the lines-detected image
		//cv::imshow("Original Image", image);
		//cv::waitKey(0);

		// Display the original image and the cropped bottom half face
		//cv::imshow("Original Image", image);
		//resizeWindow("Cropped Bottom Half Face", 200, 200);
		//cv::imshow("Cropped Bottom Half Face", croppedBottomHalfFace);
		

		////////////////////////////////////////////////////////////////
		commonBackground skinColor(faceImage);
		Vec3i skin = skinColor.findCommonColor();

		cout <<"skin color: " << skin[2] << "," << skin[1] << "," << skin[0];

		pair<int, int> top = { croppedBottomHalfFace.rows-1,0};
		pair<int, int> left = { 0,croppedBottomHalfFace.cols - 1 };
		pair<int, int> bottom = { 0,0 };

		for (int row = 0; row < croppedBottomHalfFace.rows; row++) {
			for (int col = 0; col < croppedBottomHalfFace.cols;col++) {
				Vec3b pixel = croppedBottomHalfFace.at<Vec3b>(row, col);
				int blue = pixel[0];
				int green = pixel[1];
				int red = pixel[2];
				if (
					abs(blue - skin[0]) > 256 / 4
						|| abs(green - skin[1]) > 256 / 4
						|| abs(red - skin[2]) > 256 / 4
					) {

					if (row < top.first) top.first = row;
					if (row >= bottom.first) bottom.first = row;
					if (col <= left.second) {
						left.first = row;
						left.second = col;
					}
				}
			}
		}

		cout << endl << "top: " << top.first << ", " << top.second << " left: " << left.first << ", " << left.second << " bottom: " << bottom.first << "," << bottom.second << endl;
		if (abs(top.first - left.first) < abs(bottom.first - left.first)) return "happy";
		else return "sad";
	}
	else {
		std::cout << "No face detected in the image." << std::endl;
		return "No face detected in the image.";
	}
}

int main(int argc, const char* argv[])
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

		smileCascade.load(smileCascadePath);
		faceCascade.load(haarCascadePath);
		eyeCascade.load(eyeCascadePath);

		
		VideoCapture video(0);

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

			for (Rect& face : faces)
			{
				rectangle(img, face, Scalar(50, 50, 255), 3);
				putText(img, "Face", face.tl() - Point2i(-2, 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(50, 50, 255), 2);

				Mat faceROI = imgGray(face);
				
				vector<Rect> eyes;
				eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 20, 0 | CASCADE_SCALE_IMAGE, Size(5, 5));

				vector<Rect> smiles;
				smileCascade.detectMultiScale(faceROI, smiles, 1.1, 50, 0 | CASCADE_SCALE_IMAGE, Size(80, 80));

				for (Rect& eye : eyes)
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
				/*for (Rect& smile : smiles)
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
								Vec3b& imgPixel = img.at<Vec3b>(rainbowsmilePosition.y + y, rainbowsmilePosition.x + x);
								imgPixel[0] = static_cast<uchar>((rainbowsmilePixel[3] / 255.0) * rainbowsmilePixel[0] + (1.0 - rainbowsmilePixel[3] / 255.0) * imgPixel[0]);
								imgPixel[1] = static_cast<uchar>((rainbowsmilePixel[3] / 255.0) * rainbowsmilePixel[1] + (1.0 - rainbowsmilePixel[3] / 255.0) * imgPixel[1]);
								imgPixel[2] = static_cast<uchar>((rainbowsmilePixel[3] / 255.0) * rainbowsmilePixel[2] + (1.0 - rainbowsmilePixel[3] / 255.0) * imgPixel[2]);
							}
						}
					}
				}
				*/

				Mat colorFaceRoi;
				colorFaceRoi = img(face).clone();
				//cvtColor(faceROI, colorFaceRoi, COLOR_GRAY2BGR);
				cout << detectEmotion(colorFaceRoi)<<endl;
			}

			imshow("Image", img);

			if (waitKey(50) == 27) // Break the loop if the user presses the 'Esc' key (ASCII 27)
				break;
		}

		video.release();

		/*Mat image = imread("image1.jpg");
		cout << detectEmotion(image);
		return 0;*/
	
}


