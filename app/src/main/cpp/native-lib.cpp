#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <android/log.h>

using namespace std;
using namespace cv;

extern "C"
{
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_salt(JNIEnv *env, jobject instance,
                                                                               jlong matAddrGray,
                                                                               jint nbrElem) {
        Mat &temp = *(Mat *) matAddrGray;

//        dlib::array2d<unsigned char> dlibImage;
//        dlib::assign_image(dlibImage, dlib::cv_image<unsigned char>(temp));
//        std::vector<dlib::rectangle> dets = detector(dlibImage);
        //dlib::cv_image<unsigned char> cimg(temp);

        //std::vector<dlib::rectangle> dets = detector(cimg);

        int x = 0;
        int y = 0;
        int width = 100;
        int height = 200;
        cv::Rect rect(x, y, width, height);
        cv::rectangle(temp, rect, cv::Scalar(255));//0, 255, 0));

//        for (int k = 0; k < nbrElem; k++) {
//            int i = rand() % mGr.cols;
//            int j = rand() % mGr.rows;
//            mGr.at<uchar>(j, i) = 255;
//        }
    }
}
