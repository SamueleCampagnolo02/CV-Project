// Syringe detection inference (batch mode), with per-frame timing and a
// final summary report of average inference time — used to estimate the
// maximum conveyor belt speed the system could theoretically support.
//
// Usage:
//   syringe_detector <model.onnx> <images_folder>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <algorithm>

static const std::vector<std::string> CLASS_NAMES = {
    "syringe"
};

static const int INPUT_SIZE = 640;
static const float CONF_THRESHOLD = 0.35f;
static const float NMS_IOU_THRESHOLD = 0.45f;
static const float ROI_MARGIN_RATIO = 0.05f;

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

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
    try {
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

        std::vector<double> frame_times_ms;
        cv::namedWindow("Syringe Detection", cv::WINDOW_NORMAL);

        for (const auto& image_path : image_paths) {
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Skipping unreadable file: " << image_path << std::endl;
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();

            // --- Preprocessing pipeline ---
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

            // --- Inference ---
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

            // --- Timer stops here: only preprocessing + inference + NMS counted ---
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            frame_times_ms.push_back(elapsed_ms);

            std::cout << std::filesystem::path(image_path).filename().string()
                      << " -> " << nms_indices.size() << " detections, "
                      << cv::format("%.2f", elapsed_ms) << " ms" << std::endl;

            // --- Drawing and display happen AFTER timing, not counted ---
            for (int idx : nms_indices) {
                cv::Rect box = raw_boxes[idx];
                float confidence = raw_confidences[idx];
                int class_id = raw_class_ids[idx];

                std::string label = (class_id >= 0 && class_id < static_cast<int>(CLASS_NAMES.size()))
                    ? CLASS_NAMES[class_id]
                    : "unknown";

                cv::Scalar color(0, 0, 255);
                cv::rectangle(contrast_image, box, color, 2);
                std::string text = label + " " + cv::format("%.2f", confidence);
                cv::putText(contrast_image, text, cv::Point(box.x, box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }

            cv::resizeWindow("Syringe Detection", 900, 900 * contrast_image.rows / contrast_image.cols);
            cv::imshow("Syringe Detection", contrast_image);
            std::cout << "Press any key for next image (or 'q' to quit)..." << std::endl;
            int key = cv::waitKey(0);
            if (key == 'q' || key == 'Q') break;
        }

        // --- Final timing report ---
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

            std::cout << "\n=== Inference Timing Report ===" << std::endl;
            std::cout << "Frames processed: " << frame_times_ms.size() << std::endl;
            std::cout << "Mean time:        " << cv::format("%.2f", mean) << " ms" << std::endl;
            std::cout << "Median time:      " << cv::format("%.2f", median) << " ms" << std::endl;
            std::cout << "Std deviation:    " << cv::format("%.2f", stddev) << " ms" << std::endl;
            std::cout << "Min time:         " << cv::format("%.2f", min_t) << " ms" << std::endl;
            std::cout << "Max time:         " << cv::format("%.2f", max_t) << " ms" << std::endl;
            std::cout << "Effective FPS:    " << cv::format("%.2f", 1000.0 / mean) << std::endl;
        }

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}