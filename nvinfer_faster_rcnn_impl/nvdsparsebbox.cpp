#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <map>
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

static constexpr int MAX_OUTPUT_BBOX_COUNT = 300;
static constexpr int LOCATIONS = 4;
struct alignas(float) Detection{
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Detection& a, Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; i++) 
    {
     
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

}

template<typename T>
T clamp(T x, T min, T max)
{
	return std::max(std::min(x, max), min);
}

/* This is a sample bounding box parsing function for the sample YoloV5 detector model */
static bool NvDsInferParseFasterRCNN(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{

    static int detPredLayerIndex = -1;
    static int labelPredLayerIndex = -1;
    static const int NUM_CLASSES_FASTER_RCNN = 80;
    static const int nmsMaxOut = 100;
    if (detPredLayerIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        if (strcmp(outputLayersInfo[i].layerName, "dets") == 0) {
            detPredLayerIndex = i;
            break;
        }
        }
        if (detPredLayerIndex == -1) {
        std::cerr << "Could not find bbox_pred layer buffer while parsing" << std::endl;
        return false;
        }
    }

    if (labelPredLayerIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        if (strcmp(outputLayersInfo[i].layerName, "labels") == 0) {
            labelPredLayerIndex = i;
            break;
        }
        }
        if (labelPredLayerIndex == -1) {
        std::cerr << "Could not find cls_prob layer buffer while parsing" << std::endl;
        return false;
        }
    }
    if (NUM_CLASSES_FASTER_RCNN !=
        detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        NUM_CLASSES_FASTER_RCNN << std::endl;
    }
    float *dets = (float *) outputLayersInfo[detPredLayerIndex].buffer;
    int *labels = (int *) outputLayersInfo[labelPredLayerIndex].buffer;

    for (int i = 0; i < nmsMaxOut; ++i)
    {
        // predict bbox
        float rectx1 = dets[i*5 + 0];
        float recty1 = dets[i*5 + 1];
        float rectx2 = dets[i*5 + 2];
        float recty2 = dets[i*5 + 3];
        // confidence
        float confidence = dets[i * 5 + 4];
        if (confidence < detectionParams.perClassPreclusterThreshold[0])
            continue;
        // std::cout << labels[i] << std::endl;
        int class_id = labels[i];
        NvDsInferObjectDetectionInfo object;
        object.classId = class_id;
        object.detectionConfidence = confidence;

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP(rectx1, 0, networkInfo.width - 1);
        object.top = CLIP(recty1, 0, networkInfo.height - 1);
        object.width = CLIP(rectx2, 0, networkInfo.width - 1) - object.left + 1;
        object.height = CLIP(recty2, 0, networkInfo.height - 1) - object.top + 1;

        objectList.push_back(object);

    }
    
    return true;
}

extern "C" bool NvDsInferParseCustomFasterRCNN(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseFasterRCNN(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomFasterRCNN);
