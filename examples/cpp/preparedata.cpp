#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

uint32_t fromLittleEndian(uint32_t value) {
    return (value >> 24) |
           ((value << 8) & 0x00ff0000) |
           ((value >> 8) & 0x0000ff00) |
           (value << 24);
}

void prepare_dataset(std::string &dataset){
    //std::string base_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +"_base.fvecs";
    //std::string base_out = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +".data_new";
    // std::string query_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +"_query.fvecs";
    // std::string query_out = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +".query_new";
    std::string groundtruth_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +"_groundtruth.ivecs";
    std::string groundtruth_out = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +".groundtruth_new";

    //---------------- STEP1: prepare the base input
    std::ifstream file(groundtruth_input, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }

    std::ofstream outfile(groundtruth_out, std::ios::binary);

    uint32_t d;
    file.read(reinterpret_cast<char*>(&d), sizeof(d));
    //d = fromLittleEndian(d);

    //size_t vecsizeof = sizeof(uint32_t) + d * sizeof(float); // 4 bytes for dimension + d floats
    size_t vecsizeof = sizeof(uint32_t) + d * sizeof(int); // 4 bytes for dimension + d floats

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t numVectors = fileSize / vecsizeof;

    std::cout << d << " " << numVectors << std::endl;

    int float_byte = 4;
    outfile.write(reinterpret_cast<const char*>(&float_byte), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&numVectors), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&d), sizeof(int));    

    for (uint32_t i = 0; i < numVectors; ++i) {

        uint32_t d_;
        file.read(reinterpret_cast<char*>(&d_), sizeof(d_)); 
        //std::cout << d_ << " ";   

        std::vector<float> vector(d);
        //std::vector<int> vector(d);

        // 读取向量数据
        //file.read(reinterpret_cast<char*>(vector.data()), (d) * sizeof(float));
        file.read(reinterpret_cast<char*>(vector.data()), (d) * sizeof(int));

        // 打印向量数据（或进行其他处理）
        std::cout << "Vector " << i << " with dimensionality " << d << ":\n";
        for (float value : vector) {
        //for (int value : vector) {
            outfile.write(reinterpret_cast<const char*>(&value), sizeof(float));
            //std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    //---------------- STEP2: prepare the query
    //---------------- STEP3: prepare the groundtruth

}

void prepare_dataset_openai(std::string &dataset){

    //std::string base_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + "openai_base.bin";
    std::string base_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" +"base.1B.fbin.crop_nb_100000000";
    //std::string base_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" +"base1b.fbin.crop_nb_100000000";
    std::string base_out = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +".data_new";
    //std::string base_out = "/home/data/zgongae/VectorsIndex/datasets/bigann/" + dataset +".data_new";

    std::ifstream file(base_input, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }

    std::ofstream outfile(base_out, std::ios::binary);

    uint32_t numVectors, d;
    file.read(reinterpret_cast<char*>(&numVectors), sizeof(numVectors));
    file.read(reinterpret_cast<char*>(&d), sizeof(d));


    int float_byte = 4;
    outfile.write(reinterpret_cast<const char*>(&float_byte), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&numVectors), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&d), sizeof(int));    

    std::cout << numVectors << " " << d <<std::endl;
    for (uint32_t i = 0; i < numVectors; ++i) {

        std::vector<float> vector(d);

        // 读取向量数据
        file.read(reinterpret_cast<char*>(vector.data()), (d) * sizeof(float));

        std::cout << "Vector " << i << " with dimensionality " << d << ":\n";
        for (float value : vector) {
            outfile.write(reinterpret_cast<const char*>(&value), sizeof(float));
            //std::cout << value << " ";
        }
        //std::cout << std::endl;
    }
}

int main(int argc, char** argv) {

    // std::string dataset(argv[1]);
    // if(dataset == "openai" || dataset == "sift100m" || dataset == "deep100m" || dataset == "msturing"){
    //     prepare_dataset_openai(dataset);
    // }
    // else{
    //     prepare_dataset(dataset);
    // }

    // --------------- check the query and the ground truth of "sift100m" && "deep100m"
    std::string dataset(argv[1]);
    std::string pathdeep = "/home/data/zgongae/VectorsIndex/datasets/sift100m/"+dataset;
    // pathsift = ""
    std::ifstream file_query(pathdeep, std::ios::binary);
    //std::ifstream file_truth(groundtruth_input, std::ios::binary);
    int num_query = -1;
    float **queryVectors = nullptr;
    file_query.read(reinterpret_cast<char*>(&num_query), sizeof(num_query));

    uint32_t d;
    file_query.read(reinterpret_cast<char*>(&d), sizeof(d));
    std::cout << num_query << " " << d<< std::endl;

    queryVectors = new float*[num_query];
    for (uint32_t i = 0; i < num_query; ++i) {

        queryVectors[i] = new float[d];
        file_query.read((char*)queryVectors[i], sizeof(float) * d);   
    }
    std::cout << "Finish reading ..." << std::endl; 
	for (int i = 0; i < num_query; ++i) {
		delete[] queryVectors[i];
	}
	delete[] queryVectors;  
    //

    // std::ifstream file("/home/data/zgongae/LSH-APG/dataset/glove2.2m/glove2.2m_base.fvecs", std::ios::binary);
    // if (!file.is_open()) {
    //     std::cerr << "Failed to open file." << std::endl;
    //     return 1;
    // }

    // std::ofstream outfile("/home/data/zgongae/LSH-APG/dataset/glove.data_new", std::ios::binary);

    // uint32_t d;
    // file.read(reinterpret_cast<char*>(&d), sizeof(d));
    // //d = fromLittleEndian(d);

    // size_t vecsizeof = sizeof(uint32_t) + d * sizeof(float); // 4 bytes for dimension + d floats

    // // 获取文件大小
    // file.seekg(0, std::ios::end);
    // size_t fileSize = file.tellg();
    // file.seekg(0, std::ios::beg);

    // // 计算向量数量
    // size_t numVectors = fileSize / vecsizeof;

    // std::cout << d << " " << numVectors;

    // int float_byte = 4;
    // outfile.write(reinterpret_cast<const char*>(&float_byte), sizeof(int));
    // outfile.write(reinterpret_cast<const char*>(&numVectors), sizeof(int));
    // outfile.write(reinterpret_cast<const char*>(&d), sizeof(int));

    // for (uint32_t i = 0; i < numVectors; ++i) {

    //     std::vector<float> vector(d);

    //     // 读取向量数据
    //     file.read(reinterpret_cast<char*>(vector.data()), d * sizeof(float));

    //     // 打印向量数据（或进行其他处理）
    //     std::cout << "Vector " << i << " with dimensionality " << d << ":\n";
    //     for (float value : vector) {
    //         outfile.write(reinterpret_cast<const char*>(&value), sizeof(float));
    //         //std::cout << value << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // file.close();
    // outfile.close();


}