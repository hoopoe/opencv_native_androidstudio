#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

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
        dlib::cv_image<unsigned char> cimg(temp);
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<dlib::rectangle> dets = detector(cimg);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        __android_log_print(ANDROID_LOG_INFO, "App", "Elapsed time: = %0.4f sec", elapsed.count());
        __android_log_print(ANDROID_LOG_INFO, "App", "Number of faces: = %d", dets.size());

        for(auto i : dets)
        {
            int x = i.left();
            int y = i.top();
            int width = i.width();
            int height = i.height();
            cv::Rect rect(x, y, width, height);
            cv::rectangle(temp, rect, cv::Scalar(255));
        }
    }
}
