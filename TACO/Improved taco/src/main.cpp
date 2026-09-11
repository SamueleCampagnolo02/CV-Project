// Inference per il modello TACO a 12 Macro-Categorie usando YOLO26 ONNX.
// Usage: taco_detector <model.onnx> <images_folder>

// Waste detection inference for the TACO model (12 Macro-Categories)
// using YOLO26 exported to ONNX.
//
// Usage: taco_detector <model.onnx> <images_folder>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

// 12 Macro-Categorie usate nell'addestramento Kaggle (ordine 0-11)
static const std::vector<std::string> CLASS_NAMES = {
    "Plastic_Bottles_and_Containers", // 0
    "Plastic_Films_and_Wrappers",     // 1
    "Caps_and_Lids",                  // 2
    "Metal_and_Cans",                 // 3
    "WEEE_and_Electronics",           // 4
    "Hazardous_and_Toxic",            // 5
    "Glass",                          // 6
    "Paper_and_Cardboard",            // 7
    "Styrofoam",                      // 8
    "Cigarette_Butts",                // 9
    "Organic_Waste",                  // 10
    "Other_Trash"                     // 11
};

// Categorie evidenziate in ROSSO per rilievo visivo
static bool isDangerous(const std::string& label) {
    return label == "Hazardous_and_Toxic" || label == "Glass" || label == "WEEE_and_Electronics";
}

static const int INPUT_SIZE = 640;
static const float CONF_THRESHOLD = 0.20f; // Soglia confidenza (20%)
static const float NMS_IOU_THRESHOLD = 0.45f;

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

// Ridimensiona l'immagine a INPUT_SIZE x INPUT_SIZE mantenendo l'aspect ratio (letterboxing)
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

    // Caricamento rete ONNX
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
    if (net.empty()) {
        std::cerr << "Error: could not load ONNX model at " << model_path << std::endl;
        return 1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Lettura delle immagini dalla cartella
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

    cv::namedWindow("TACO Waste Detection", cv::WINDOW_NORMAL);

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

        // --- Inference Sicura ---
        cv::Mat output;
        try {
            output = net.forward();
        } catch (const cv::Exception& e) {
            std::cerr << "❌ Errore durante net.forward(): " << e.what() << std::endl;
            continue;
        }

        // Conversione sicura da tensore 3D [1, 300, 6] a matrice 2D [300, 6] senza usare .reshape()
        cv::Mat detections;
        if (output.dims == 3) {
            detections = cv::Mat(output.size[1], output.size[2], CV_32F, const_cast<float*>(output.ptr<float>(0)));
        } else if (output.dims == 2) {
            detections = output;
        } else {
            std::cerr << "Formato output non supportato (dims = " << output.dims << ")" << std::endl;
            continue;
        }

        std::vector<cv::Rect> raw_boxes;
        std::vector<float> raw_confidences;
        std::vector<int> raw_class_ids;

        // Estrazione risultati
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

            // Mappatura coordinate sull'immagine originale
            float orig_x1 = (x1 - pad_x) / scale;
            float orig_y1 = (y1 - pad_y) / scale;
            float orig_x2 = (x2 - pad_x) / scale;
            float orig_y2 = (y2 - pad_y) / scale;

            cv::Rect box(
                cv::Point(static_cast<int>(orig_x1), static_cast<int>(orig_y1)),
                cv::Point(static_cast<int>(orig_x2), static_cast<int>(orig_y2))
            );
            box &= cv::Rect(0, 0, image.cols, image.rows); // Ritaglio entro i bordi dell'immagine

            raw_boxes.push_back(box);
            raw_confidences.push_back(confidence);
            raw_class_ids.push_back(class_id);
        }

        // --- Non-Maximum Suppression (NMS) ---
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(raw_boxes, raw_confidences, CONF_THRESHOLD, NMS_IOU_THRESHOLD, nms_indices);

        std::vector<Detection> results;
        for (int idx : nms_indices) {
            results.push_back({raw_boxes[idx], raw_confidences[idx], raw_class_ids[idx]});
        }

        // --- Disegno a schermo e Log ---
        std::cout << "\n=== " << std::filesystem::path(image_path).filename().string() << " ===" << std::endl;
        for (const auto& det : results) {
            std::string label = (det.class_id >= 0 && det.class_id < static_cast<int>(CLASS_NAMES.size()))
                ? CLASS_NAMES[det.class_id]
                : "Unknown";

            cv::Scalar color = isDangerous(label) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

            // Disegno del rettangolo
            cv::rectangle(image, det.box, color, 2);
            
            // Disegno del testo con sfondo pieno per migliore leggibilita
            std::string text = label + " (" + cv::format("%.2f", det.confidence) + ")";
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            
            cv::Point textOrg(det.box.x, std::max(det.box.y - 5, textSize.height));
            cv::rectangle(image, cv::Point(textOrg.x, textOrg.y - textSize.height - 2),
                          cv::Point(textOrg.x + textSize.width, textOrg.y + baseline), color, cv::FILLED);
            
            cv::putText(image, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

            std::cout << " -> " << label << " [" << cv::format("%.2f", det.confidence) << "]" << std::endl;
        }

        if (results.empty()) {
            std::cout << " Nessun rifiuto rilevato." << std::endl;
        }

        // Visualizzazione finestra
        cv::resizeWindow("TACO Waste Detection", 900, 900 * image.rows / image.cols);
        cv::imshow("TACO Waste Detection", image);
        
        std::cout << "Premere un tasto qualsiasi per la foto successiva ('q' per uscire)..." << std::endl;
        int key = cv::waitKey(0);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }

    return 0;
}