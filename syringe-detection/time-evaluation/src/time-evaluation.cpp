#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <algorithm>
static const int INPUT_SIZE = 640;
static const float CONF_THRESHOLD = 0.35f;
static const float NMS_IOU_THRESHOLD = 0.45f;
static const float ROI_MARGIN_RATIO = 0.05f;
cv::Mat cropROI(const cv::Mat& src, float margin_ratio = ROI_MARGIN_RATIO) {
    int margin_x = static_cast<int>(src.cols * margin_ratio);
    int margin_y = static_cast<int>(src.rows * margin_ratio);
    cv::Rect roi(margin_x, margin_y, src.cols - 2 * margin_x, src.rows - 2 * margin_y);
    return src(roi).clone();
}
cv::Mat applyCLAHE(const cv::Mat& src) {
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab, lab_planes);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    cv::merge(lab_planes, lab);
    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}
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
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp") {
            image_paths.push_back(entry.path().string());
        }
    }
    if (image_paths.empty()) {
        std::cerr << "No images found in " << folder_path << std::endl;
        return 1;
    }
    std::vector<double> frame_times_ms;
    int total_detections = 0;
    for (const auto& image_path : image_paths) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Skipping unreadable file: " << image_path << std::endl;
            continue;
        }
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat roi_image = cropROI(image);
        cv::Mat contrast_image = applyCLAHE(roi_image);
        float scale;
        int pad_x, pad_y;
        cv::Mat letterboxed = letterbox(contrast_image, scale, pad_x, pad_y);
        cv::Mat blob = cv::dnn::blobFromImage(
            letterboxed, 1.0 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE),
            cv::Scalar(), true, false
        );
        net.setInput(blob);
        cv::Mat output = net.forward();
        int num_predictions = output.size[2];
        int num_channels = output.size[1];
        int num_classes = num_channels - 4;
        cv::Mat raw(num_channels, num_predictions, CV_32F, output.ptr<float>());
        cv::Mat transposed;
        cv::transpose(raw, transposed);
        std::vector<cv::Rect> raw_boxes;
        std::vector<float> raw_confidences;
        std::vector<int> raw_class_ids;
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
            if (best_score < CONF_THRESHOLD) continue;
            float x1 = (cx - w / 2 - pad_x) / scale;
            float y1 = (cy - h / 2 - pad_y) / scale;
            float box_w = w / scale;
            float box_h = h / scale;
            cv::Rect box(static_cast<int>(x1), static_cast<int>(y1),
                          static_cast<int>(box_w), static_cast<int>(box_h));
            box &= cv::Rect(0, 0, contrast_image.cols, contrast_image.rows);
            raw_boxes.push_back(box);
            raw_confidences.push_back(best_score);
            raw_class_ids.push_back(best_class);
        }
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(raw_boxes, raw_confidences, CONF_THRESHOLD, NMS_IOU_THRESHOLD, nms_indices);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        frame_times_ms.push_back(elapsed_ms);
        total_detections += static_cast<int>(nms_indices.size());
        std::cout << std::filesystem::path(image_path).filename().string()
                  << " -> " << nms_indices.size() << " detections, "
                  << cv::format("%.2f", elapsed_ms) << " ms" << std::endl;
    }
    if (!frame_times_ms.empty()) {
        double sum = std::accumulate(frame_times_ms.begin(), frame_times_ms.end(), 0.0);
        double mean = sum / frame_times_ms.size();
        double sq_sum = 0.0;
        for (double t : frame_times_ms) sq_sum += (t - mean) * (t - mean);
        double stddev = std::sqrt(sq_sum / frame_times_ms.size());
        double min_t = *std::min_element(frame_times_ms.begin(), frame_times_ms.end());
        double max_t = *std::max_element(frame_times_ms.begin(), frame_times_ms.end());
        std::vector<double> sorted_times = frame_times_ms;
        std::sort(sorted_times.begin(), sorted_times.end());
        double median = sorted_times[sorted_times.size() / 2];
        std::ostringstream report;
        report << "=== Inference Timing Report ===\n";
        report << "Frames processed: " << frame_times_ms.size() << "\n";
        report << "Total detections: " << total_detections << "\n";
        report << "Mean time:        " << cv::format("%.2f", mean) << " ms\n";
        report << "Median time:      " << cv::format("%.2f", median) << " ms\n";
        report << "Std deviation:    " << cv::format("%.2f", stddev) << " ms\n";
        report << "Min time:         " << cv::format("%.2f", min_t) << " ms\n";
        report << "Max time:         " << cv::format("%.2f", max_t) << " ms\n";
        report << "Effective FPS:    " << cv::format("%.2f", 1000.0 / mean) << "\n";

        std::cout << "\n" << report.str();

        std::ofstream out_file("time-evaluation.txt");
        out_file << report.str();
        out_file.close();
        std::cout << "\nReport saved to time-evaluation.txt" << std::endl;
    }
    return 0;
}
