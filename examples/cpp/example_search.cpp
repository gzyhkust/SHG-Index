#include "../../hnswlib/hnswlib.h"

int main(int argc, char** argv) {


    // ------------------------------  STEP 1: Loading the data ------------------------------  
    int dim = -1;               // Dimension of the elements
    int num_base = -1;   // Maximum number of elements, should be known beforehand
    int M = 48;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 80;  // Controls index search speed/build speed tradeoff

    std::string algo(argv[1]);
    std::string dataset(argv[2]);
    int K = std::stoi(argv[3]); 

    //std::string file = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +".data_new";
    std::string file = "/home/data/zgongae/VectorsIndex/HEDS/enron/" + dataset +".data_new";
    std::ifstream loadin(file.c_str(), std::ios::binary);
    while (!loadin) {
        printf("Fail to find data file!\n");
        exit(0);
    }

    unsigned int header[3] = {};
    loadin.read((char*)header, sizeof(header));

    int num_query = -1;
    float **queryVectors = nullptr;
    int **groundtruth = nullptr;

    if(dataset == "openai" || dataset == "deep10m" || dataset == "msturing" || dataset == "sift10m"){
        load_query_groundtruth_(dataset, num_query, queryVectors, groundtruth);
    }else{
        load_query_groundtruth(dataset, num_query, queryVectors, groundtruth);
    }

    // ------------------------------  STEP 2: Initial the index ------------------------------ 
    num_base = header[1];
    dim = header[2];

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HEDS<float>* alg_hnsw = new hnswlib::HEDS<float>(&space, dim, num_base + num_query, M, ef_construction);
    std::cout << alg_hnsw -> max_elements_ << " " << num_base << " " <<  num_query << std::endl; 
    // ------------------------------  STEP 3: Build the index ------------------------------ 
    Performance per;
    Timer t;
    
    float** data = new float* [num_base];
    for(int i = 0; i< num_base; ++i){
        data[i] = new float[dim];
        loadin.read((char*)data[i], sizeof(float) * header[2]);        
    }


    t.restart();
    for(int i = 0; i< num_base; ++i){
        if(algo == "heds"){
            alg_hnsw->addDataPoint(data[i], i, 1, per);
        }else{
            alg_hnsw->addPointHNSW(data[i], i, per);
        }     
    }
    per.setTimeBuildindex(t.elapsed());

    //std::string construction_res_path = "/home/data/zgongae/VectorsIndex/HEDS/results/" + dataset + "_" + algo +"_construction.txt";
    //std::ofstream construction_res(construction_res_path);

    std::cout << "Build Index Time : " << per.getTimeBuildindex() << " [s]" << std::endl;
    //construction_res << "Build Index Time : " << per.getTimeBuildindex() << " [s]" << std::endl;
    std::cout << "Preprocessing Time: " << per.getTimePreprocessing() << " [s]" << std::endl;
    //construction_res << "Preprocessing Time: " << per.getTimePreprocessing() << " [s]" << std::endl;

    t.restart();
    if(algo == "heds"){
        alg_hnsw->buildShortcuts(num_base);
    }
    per.setTimeShortcut(t.elapsed());

    // std::cout << "Tree height: " << alg_hnsw->maxlevel_ << std::endl;
    // std::cout << "Num of elements: " << num_base << std::endl;
    //std::cout << "Build shortcut time: " << per.getTimeShortcut() << std::endl;
    //construction_res << "Build shortcut time: " << per.getTimeShortcut() << std::endl;
    //std::cout << "Memory cost: " << (alg_hnsw->indexFileSize(num_base))/ (1024.0 * 1024.0) << std::endl;
    //construction_res << "Memory cost: " << (alg_hnsw->indexFileSize(num_base))/ (1024.0 * 1024.0) << std::endl;
    //construction_res.close();
    // ------------------------------  STEP 3: Load the query and groundtruth ------------------------------
    // std::string search_res_path = "/home/data/zgongae/VectorsIndex/HEDS/results_varyK/" + dataset + "_" + algo + "_"+std::to_string(K)+"_search.txt";
    // std::ofstream search_res(search_res_path);

    for(int i = 0; i< num_query; i++){
        alg_hnsw->addDataPoint(queryVectors[i], num_base+i, -1, per);
        Query query(num_base+i, K);
        std::cout << "query:" << i << std::endl;
        t.restart();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
        if(algo == "heds"){
            result = alg_hnsw->searchKnnShortcuts(query);
        }else{
            result = alg_hnsw->searchKnn(queryVectors[i], query.getID(),K);
        }
        query.setQueryTime(t.elapsed());
        query.setRecall(calculateRecall(i, K, result, groundtruth));

        std:: cout << i << ": time: " << query.getQueryTime() << " Recall: " << query.getRecall() << std::endl;
        //search_res << query.getQueryTime() << " " << query.getRecall() << std::endl;
    }
    // search_res.close();

    clear_2d_array(queryVectors, num_query);
    clear_2d_array(groundtruth, num_query);
    clear_2d_array(data, num_base);
    delete alg_hnsw;
    return 0;
}