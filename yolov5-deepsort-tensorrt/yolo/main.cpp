#include <dirent.h>
#include "yolo_lib.h"


int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names)
{
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


int run(char* imageDir)
{
    // get image file names for inferencing
    std::vector<std::string> file_names;
    if (read_files_in_dir(imageDir, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // create detecter, and load engine plan
    std::string trtFile = "../resources/model.plan";
    int gpuId = 0;
    YoloDetecter detecter(trtFile, gpuId);

    // inference
    for (long unsigned int i = 0; i < file_names.size(); i++)
    {
        std::string imagePath = std::string(imageDir) + "/" + file_names[i];
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        std::vector<DetectResult> res = detecter.inference(img);

        // draw result on image
        for (long unsigned int j = 0; j < res.size(); j++)
        {
            cv::Rect r = res[j].tlwh;
            cv::rectangle(img, r, cv::Scalar(255, 0, 255), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
        }

        cv::imwrite("_" + file_names[i], img);

        std::cout << "Image: " << file_names[i] << " done." << std::endl;
    }

    return 0;
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("This program need 1 argument\n");
        printf("Usage: ./yolo_infer [image dir]\n");
        printf("Example: ./yolo_infer ./images\n");
        return 1;
    }

    return run(argv[1]);
}
