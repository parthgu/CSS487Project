// main.cpp
// This code demonstrates practice of opencv syntax by overlay background image and edge detect image
// Author: Dong Nguyen

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "colorHistogram.cpp"
#include <opencv2/imgproc.hpp>
using namespace cv;

class commonBackground
{
public:
    Mat foreImage;
    commonBackground(Mat &foreImage)
    {
        this->foreImage = foreImage.clone();
    }

    // findMostVotedBin - finding the bin that has most vote in the input histogram
    // precondition: hist is valid 3 dimension diagram
    // postcondition: Return the most voted bin
    Vec3i findMostVotedBin(const Mat &hist)
    {
        int max_count = 0;
        Vec3i most_voted_bin;
        // To store the bin indices with the maximum count
        for (int blueDimension = 0; blueDimension < hist.size[0]; blueDimension++)
        {
            for (int greenDimension = 0; greenDimension < hist.size[1]; greenDimension++)
            {
                for (int redDimension = 0; redDimension < hist.size[2]; redDimension++)
                {
                    int bin_count = hist.at<int>(blueDimension, greenDimension, redDimension);
                    if (bin_count > max_count)
                    {
                        max_count = bin_count;
                        most_voted_bin = Vec3i(blueDimension, greenDimension, redDimension); // Reversed order to match BGR
                    }
                }
            }
        }

        return most_voted_bin;
    }

    // main - exercise overlay background image and edge detection image
    // precondition: foreground.jpg exists and background.jpg in the code directory and is a valid JPG
    // postconditions: The original and sharpened images are displayed on the screen
    //					Create the image that is overlay version of foreground.jpg by background.jpg
    //					Display that image on the screen and written it to the disk as overlay.jpg
    //					Also horizontally flip, convert to gray scale, smooth and edge detect background.jpg
    //					Display the result after edge detect on the screen and write it on disk as output.jpg
    //					Also create a resize and rotate 90 degree in clockwise version of foreground.jpg
    //					Display the resize, rotate version on the screen and write it on disk as myoutput.jpg
    //					Note: waits for a key press between each image display

    Vec3i findCommonColor()
    {
        // Mat foreImage = imread("foreground.jpg");
        // Mat backgroundImage = imread("background.jpg");

        int bucketSize = 256 / 4;
        int dims[] = {4, 4, 4};
        Mat hist(3, dims, CV_32S, Scalar::all(0));

        for (int row = 0; row < foreImage.rows; row++)
        {
            for (int col = 0; col < foreImage.cols; col++)
            {

                Vec3b pixel = foreImage.at<Vec3b>(row, col);
                int r = pixel[2] / bucketSize;
                int g = pixel[1] / bucketSize;
                int b = pixel[0] / bucketSize;

                hist.at<int>(b, g, r) += 1;
            }
        }

        Vec3i most_voted_bin = findMostVotedBin(hist);

        int cRed = most_voted_bin[2] * bucketSize + bucketSize / 2;
        int cGreen = most_voted_bin[1] * bucketSize + bucketSize / 2;
        int cBlue = most_voted_bin[0] * bucketSize + bucketSize / 2;

        Vec3i binColor = Vec3i(cBlue, cGreen, cRed);

        return binColor;
    }
};
