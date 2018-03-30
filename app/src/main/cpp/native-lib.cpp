#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/svm/svm_multiclass_linear_trainer.h>

#include <android/log.h>

using namespace std;
using namespace cv;
using namespace dlib;

#define AppTag "OCVSample::Activity"

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
 alevel0<
         alevel1<
                 alevel2<
                         alevel3<
                                 alevel4<
                                         dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                                                 dlib::input_rgb_image_sized<150>
                         >>>>>>>>>>>>;

dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
dlib::shape_predictor sp;
anet_type net;
matrix<float, 0, 1> first_face;
matrix<float, 0, 1> second_face;

typedef matrix<float, 0, 1> sample_type;
typedef linear_kernel<sample_type> lin_kernel;
multiclass_linear_decision_function<lin_kernel, string> df;

bool resourceLoaded;

extern "C"
{
    void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_loadResources(JNIEnv *env, jobject instance) {
        __android_log_print(ANDROID_LOG_DEBUG, AppTag, "Load resources started");
        FILE* file = fopen("/storage/emulated/0/Movies/shape_predictor_5_face_landmarks.dat","r+");
        FILE* file2 = fopen("/storage/emulated/0/Movies/dlib_face_recognition_resnet_model_v1.dat","r+");
        FILE* file3 = fopen("/storage/emulated/0/Movies/first_face_feature_vector.dat","r+");
        FILE* file4 = fopen("/storage/emulated/0/Movies/second_face_feature_vector.dat","r+");
        FILE* file5 = fopen("/storage/emulated/0/Movies/faces_linear.svm","r+");

        if (file != NULL && file2 != NULL && file3 != NULL && file4 != NULL && file5 != NULL)
        {
            dlib::deserialize("/storage/emulated/0/Movies/shape_predictor_5_face_landmarks.dat") >> sp;
            dlib::deserialize("/storage/emulated/0/Movies/dlib_face_recognition_resnet_model_v1.dat") >> net;
            dlib::deserialize("/storage/emulated/0/Movies/first_face_feature_vector.dat") >> first_face;
            dlib::deserialize("/storage/emulated/0/Movies/second_face_feature_vector.dat") >> second_face;
            dlib::deserialize("/storage/emulated/0/Movies/faces_linear.svm") >> df;
            resourceLoaded = true;
            __android_log_print(ANDROID_LOG_DEBUG, AppTag, "Resources found");
        } else{
            resourceLoaded = false;
            __android_log_print(ANDROID_LOG_DEBUG, AppTag, "Resources NOT found");
        }
        __android_log_print(ANDROID_LOG_DEBUG, AppTag, "Load resources completed");
    }

    void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_salt(JNIEnv *env, jobject instance,
                                                                               jlong matAddrGray,
                                                                               jint nbrElem) {
        Mat &temp = *(Mat *) matAddrGray;
        std::vector<matrix<rgb_pixel>> faces;
        dlib::cv_image<unsigned char> cimg(temp);
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<dlib::rectangle> dets = detector(cimg);

        for(auto face : dets)
        {
            int x = face.left();
            int y = face.top();
            int width = face.width();
            int height = face.height();
            cv::Rect rect(x, y, width, height);
            cv::rectangle(temp, rect, cv::Scalar(255));

            if (resourceLoaded) {
                auto shape = sp(cimg, face);
                matrix<rgb_pixel> face_chip;
                extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(move(face_chip));
            }
        }

        if (resourceLoaded) {
            std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
            for (size_t i = 0; i < face_descriptors.size(); ++i) {
                float dist_first = length(face_descriptors[i] - first_face);
                float dist_second = length(face_descriptors[i] - second_face);
                if (dist_first < 0.6) {
                    putText( temp, "First", Point2f(100,100), FONT_HERSHEY_PLAIN, 2,  Scalar(255));
                    std::pair<string, float> res = df.predict(face_descriptors[i]);
                    __android_log_print(ANDROID_LOG_INFO, AppTag, "label: %s, prob: = %0.2f ",
                                        res.first.c_str(), res.second);
                } else if( dist_second < 0.6) {
                    putText( temp, "Second", Point2f(100,100), FONT_HERSHEY_PLAIN, 2,  Scalar(255));
                } else {
                    putText( temp, "Unknown", Point2f(100,100), FONT_HERSHEY_PLAIN, 2,  Scalar(255));
                }
            }
        }

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        __android_log_print(ANDROID_LOG_INFO, AppTag, "Elapsed time: = %0.2f sec", elapsed.count());
    }
}
