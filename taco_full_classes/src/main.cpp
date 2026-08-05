// Waste detection inference for the TACO baseline model (60 original
// classes, no macro-grouping), using YOLO26 exported to ONNX.
//
// YOLO26's end-to-end head already filters most duplicates internally,
// but with a weak model (low mAP) overlapping boxes can still appear,
// so NMS is applied here as an extra safety filter: among overlapping
// boxes, only the highest-confidence one is kept.
//
// Usage:
//   taco_detector <model.onnx> <images_folder>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

// Class names must match the exact order in the original TACO
// annotations.json (60 categories, no grouping).
static const std::vector<std::string> CLASS_NAMES = {
    "Aluminium foil",
    "Battery",
    "Aluminium blister pack",
    "Carded blister pack",
    "Other plastic bottle",
    "Clear plastic bottle",
    "Glass bottle",
    "Plastic bottle cap",
    "Metal bottle cap",
    "Broken glass",
    "Food Can",
    "Aerosol",
    "Drink can",
    "Toilet tube",
    "Other carton",
    "Egg carton",
    "Drink carton",
    "Corrugated carton",
    "Meal carton",
    "Pizza box",
    "Paper cup",
    "Disposable plastic cup",
    "Foam cup",
    "Glass cup",
    "Other plastic cup",
    "Food waste",
    "Glass jar",
    "Plastic lid",
    "Metal lid",
    "Other plastic",
    "Magazine paper",
    "Tissues",
    "Wrapping paper",
    "Normal paper",
    "Paper bag",
    "Plastified paper bag",
    "Plastic film",
    "Six pack rings",
    "Garbage bag",
    "Other plastic wrapper",
    "Single-use carrier bag",
    "Polypropylene bag",
    "Crisp packet",
    "Spread tub",
    "Tupperware",
    "Disposable food container",
    "Foam food container",
    "Other plastic container",
    "Plastic glooves",
    "Plastic utensils",
    "Pop tab",
    "Rope & strings",
    "Scrap metal",
    "Shoe",
    "Squeezable tube",
    "Plastic straw",
    "Paper straw",
    "Styrofoam piece",
    "Unlabeled litter",
    "Cigarette"
};

// A few categories are treated as safety-relevant for display purposes
// (drawn in red instead of green).
static bool isDangerous(const std::string& label) {
    return label == "Battery" || label == "Broken glass";
}

static const int INPUT_SIZE = 640;
static const float CONF_THRESHOLD = 0.05f;
static const float NMS_IOU_THRESHOLD = 0.45f;

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

// Resizes the image to INPUT_SIZE x INPUT_SIZE while preserving aspect
// ratio, padding with gray borders (letterboxing). Returns the scale
// factor and padding offsets, needed later to map boxes back to the
// original image coordinates.
cv::Mat letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y) {
    int w = src.cols;
    int h = src.rows;

    scale = std::min(static_cast<float>(INPUT_SIZE) / w, static_cast<float>(INPUT_SIZE) / h);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    pad_x = (INPUT_SIZE - new_w) / 2;
    pad_y = (INPUT_SIZE - new_h) / 2;

    cv::Mat output(INPUT_SIZE, INPUT_SIZE, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(output(cv::Rect(pad_x, pad_y, new_w, new_h)));

    return output;
}

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <images_folder>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string folder_path = argv[2];

    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
    if (net.empty()) {
        std::cerr << "Error: could not load ONNX model at " << model_path << std::endl;
        return 1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::vector<std::string> image_paths;
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        for (auto& c : ext) c = std::tolower(c);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            image_paths.push_back(entry.path().string());
        }
    }

    if (image_paths.empty()) {
        std::cerr << "No images found in " << folder_path << std::endl;
        return 1;
    }

    cv::namedWindow("TACO Detection", cv::WINDOW_NORMAL);

    for (const auto& image_path : image_paths) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Skipping unreadable file: " << image_path << std::endl;
            continue;
        }

        // --- Preprocessing ---
        float scale;
        int pad_x, pad_y;
        cv::Mat letterboxed = letterbox(image, scale, pad_x, pad_y);

        cv::Mat blob = cv::dnn::blobFromImage(
            letterboxed, 1.0 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE),
            cv::Scalar(), true, false
        );
        net.setInput(blob);

        // --- Inference ---
        cv::Mat output = net.forward();
        // Expected shape: [1, 300, 6] -> reshape to [300, 6] for easy access
        cv::Mat detections = output.reshape(1, output.size[1]);

        // --- Collect raw candidate detections above the confidence threshold ---
        std::vector<cv::Rect> raw_boxes;
        std::vector<float> raw_confidences;
        std::vector<int> raw_class_ids;

        for (int i = 0; i < detections.rows; ++i) {
            float x1 = detections.at<float>(i, 0);
            float y1 = detections.at<float>(i, 1);
            float x2 = detections.at<float>(i, 2);
            float y2 = detections.at<float>(i, 3);
            float confidence = detections.at<float>(i, 4);
            int class_id = static_cast<int>(detections.at<float>(i, 5));

            if (confidence < CONF_THRESHOLD) {
                continue;
            }

            // Undo the letterbox transform to map back to original image coordinates
            float orig_x1 = (x1 - pad_x) / scale;
            float orig_y1 = (y1 - pad_y) / scale;
            float orig_x2 = (x2 - pad_x) / scale;
            float orig_y2 = (y2 - pad_y) / scale;

            cv::Rect box(
                cv::Point(static_cast<int>(orig_x1), static_cast<int>(orig_y1)),
                cv::Point(static_cast<int>(orig_x2), static_cast<int>(orig_y2))
            );
            box &= cv::Rect(0, 0, image.cols, image.rows); // clip to image bounds

            raw_boxes.push_back(box);
            raw_confidences.push_back(confidence);
            raw_class_ids.push_back(class_id);
        }

        // --- Non-Maximum Suppression ---
        // Among overlapping boxes (IoU > NMS_IOU_THRESHOLD), keep only the
        // one with the highest confidence.
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(raw_boxes, raw_confidences, CONF_THRESHOLD, NMS_IOU_THRESHOLD, nms_indices);

        std::vector<Detection> results;
        for (int idx : nms_indices) {
            results.push_back({raw_boxes[idx], raw_confidences[idx], raw_class_ids[idx]});
        }

        // --- Draw + print ---
        std::cout << "--- " << std::filesystem::path(image_path).filename().string() << " ---" << std::endl;
        for (const auto& det : results) {
            std::string label = (det.class_id >= 0 && det.class_id < static_cast<int>(CLASS_NAMES.size()))
                ? CLASS_NAMES[det.class_id]
                : "unknown";

            cv::Scalar color = isDangerous(label) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

            cv::rectangle(image, det.box, color, 2);
            std::string text = label + " " + cv::format("%.2f", det.confidence);
            cv::putText(image, text, cv::Point(det.box.x, det.box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

            std::cout << label << " (" << cv::format("%.2f", det.confidence) << ")" << std::endl;
        }
        if (results.empty()) {
            std::cout << "No objects detected." << std::endl;
        }

        cv::resizeWindow("TACO Detection", 900, 900 * image.rows / image.cols);
        cv::imshow("TACO Detection", image);
        std::cout << "Press any key for next image (or 'q' to quit)..." << std::endl;
        int key = cv::waitKey(0);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }

    return 0;
}