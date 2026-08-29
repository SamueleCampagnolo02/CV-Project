// Syringe detection inference using the dedicated YOLO26 model exported
// to ONNX (end2end=False, traditional head with manual NMS — the
// end-to-end NMS-free export hits an OpenCV DNN TopK limitation on
// this model, so we fall back to the classic approach here).
//
// Usage:
//   syringe_detector <model.onnx> <images_folder>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

static const std::vector<std::string> CLASS_NAMES = {
    "syringe"
};

static const int INPUT_SIZE = 640;
static const float CONF_THRESHOLD = 0.35f;

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

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

    cv::namedWindow("Syringe Detection", cv::WINDOW_NORMAL);

    for (const auto& image_path : image_paths) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Skipping unreadable file: " << image_path << std::endl;
            continue;
        }

        float scale;
        int pad_x, pad_y;
        cv::Mat letterboxed = letterbox(image, scale, pad_x, pad_y);

        cv::Mat blob = cv::dnn::blobFromImage(
            letterboxed, 1.0 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE),
            cv::Scalar(), true, false
        );
        net.setInput(blob);

        // Output shape with end2end=False: (1, 4+num_classes, num_predictions)
        cv::Mat output = net.forward();
        int num_predictions = output.size[2];
        int num_channels = output.size[1];
        int num_classes = num_channels - 4;

        cv::Mat raw(num_channels, num_predictions, CV_32F, output.ptr<float>());
        cv::Mat transposed;
        cv::transpose(raw, transposed);

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < transposed.rows; ++i) {
            float cx = transposed.at<float>(i, 0);
            float cy = transposed.at<float>(i, 1);
            float w = transposed.at<float>(i, 2);
            float h = transposed.at<float>(i, 3);

            int best_class = 0;
            float best_score = transposed.at<float>(i, 4);
            for (int c = 1; c < num_classes; ++c) {
                float score = transposed.at<float>(i, 4 + c);
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            if (best_score < CONF_THRESHOLD) {
                continue;
            }

            float x1 = (cx - w / 2 - pad_x) / scale;
            float y1 = (cy - h / 2 - pad_y) / scale;
            float box_w = w / scale;
            float box_h = h / scale;

            boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                                static_cast<int>(box_w), static_cast<int>(box_h));
            confidences.push_back(best_score);
            class_ids.push_back(best_class);
        }

        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.45f, nms_indices);

        std::vector<Detection> results;
        for (int idx : nms_indices) {
            cv::Rect box = boxes[idx] & cv::Rect(0, 0, image.cols, image.rows);
            results.push_back({box, confidences[idx], class_ids[idx]});
        }

        std::cout << "--- " << std::filesystem::path(image_path).filename().string() << " ---" << std::endl;
        for (const auto& det : results) {
            std::string label = (det.class_id >= 0 && det.class_id < static_cast<int>(CLASS_NAMES.size()))
                ? CLASS_NAMES[det.class_id]
                : "unknown";

            cv::Scalar color(0, 0, 255);

            cv::rectangle(image, det.box, color, 2);
            std::string text = label + " " + cv::format("%.2f", det.confidence);
            cv::putText(image, text, cv::Point(det.box.x, det.box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

            std::cout << label << " (" << cv::format("%.2f", det.confidence) << ")" << std::endl;
        }
        if (results.empty()) {
            std::cout << "No objects detected." << std::endl;
        }

        cv::resizeWindow("Syringe Detection", 900, 900 * image.rows / image.cols);
        cv::imshow("Syringe Detection", image);
        std::cout << "Press any key for next image (or 'q' to quit)..." << std::endl;
        int key = cv::waitKey(0);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }

    return 0;
}