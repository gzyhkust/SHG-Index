#include "../../hnswlib/testing/hnswlib.h"
#include <random>
#include <typeinfo>
#include <time.h>


void pointsKmeans(hnswlib::HierarchicalNSW<float>* alg_hnsw, int dim, int max_elements, int k, std::vector<std::vector<int>>& clusters){
    int maxIterations = 5;
    
    // std::vector<int> centroids;
    clusters.resize(k);
    //centroids.resize(k);
    std::vector<std::vector<int>> avg;

    // free clusters
    //for(int cluID = 0; cluID<k; cluID++)  clusters[cluID].swap(std::vector<std::vector<int>>());
    //for(int cluID = 0; cluID<k; cluID++)  clusters[cluID].clear();

    // find centers for clusters
    for(int iteration = 0; iteration<maxIterations; iteration++){
        //centroids.clear();
        //centroids.swap(std::vector<int>());
        std::cout << " Iteration  " << iteration << std::endl;
        std::vector<int> centroids;
        if(iteration == 0){
            for(int i = 0; i<k; i++){
                centroids.push_back(rand()%max_elements);
                //std::cout<<centroids[i] << std::endl;
            }
        }else{
            for(int cluID = 0; cluID<k; cluID++)  clusters[cluID].clear();
            for(int cluID = 0; cluID<k; cluID++){
                double dis = 65536;
                int res = 0;
                int pointID_= -1;
                for(int pointID = 0; pointID<max_elements; pointID++){
                    char *pointData = (alg_hnsw->getDataByInternalId(pointID));
                    for(int k =0;k<dim;k++)
                    {
                        res+= ((*pointData) - (avg[cluID][k])) * ((*pointData) - (avg[cluID][k]));
                        pointData++;
                    }
                    if(res<dis)
                    {
                        dis = res; 
                        pointID_  =pointID;
                    }
                }
                centroids.push_back(pointID_);
            }
        }

        for(int pointID = 0; pointID<max_elements; pointID++){
            double dis = 65536;
            int c = -1;
            for(int cluID = 0; cluID<k; cluID++){
                alg_hnsw->getDataByInternalId(pointID);
                //std::cout <<cluID << " 111111" <<std::endl;
                alg_hnsw->getDataByInternalId(centroids[cluID]);
                //std::cout <<cluID <<  " 222222" <<std::endl;
                double curdist = alg_hnsw->fstdistfunc_(alg_hnsw->getDataByInternalId(pointID),
                                alg_hnsw->getDataByInternalId(centroids[cluID]),
                                alg_hnsw->dist_func_param_);
                //std::cout<< curdist << std::endl;
                if(curdist < dis){
                    dis = curdist;
                    c = cluID;
                }
            }
            //std::cout<< c << " " << pointID << std::endl;
            clusters[c].push_back(pointID);
        }

        // update the avg for each iteration
        std::cout<< "avg: " <<std::endl;
        avg.resize(k);
        if(iteration == maxIterations) return;
        for(int cluID = 0; cluID<k; cluID++){
            // avg
            avg[cluID].resize(dim);
            for (int i = 0; i<clusters[cluID].size();i++){
                char *pointData = (alg_hnsw->getDataByInternalId(clusters[cluID][i]));
                for(int j = 0; j<dim; j++){
                    if(i == 0){
                        avg[cluID][j] = int (* pointData);
                        pointData++;
                    }else{
                        avg[cluID][j] = (avg[cluID][j] + int (* pointData)) /(j+1);
                        pointData++;
                    }
                }
            }
        }
    }
}

int main() {

    /////////////////////////////////////// BUILDING INDEX PROCESSING //////////////////////////////////
    // Step 1: Loading the data
    int dim = -1;               // Dimension of the elements
    int max_elements = -1;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    std::string file = "/home/data/zgongae/audio.data_new";
    std::ifstream loadin(file.c_str(), std::ios::binary);
    while (!loadin) {
        printf("Fail to find data file!\n");
        exit(0);
    }

    unsigned int header[3] = {};
    loadin.read((char*)header, sizeof(header));

    // Step 2: Initial the index
    max_elements = header[1];
    dim = header[2];

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    clock_t start,end;
    start = clock();  
    float** data = new float* [max_elements];
    for(int i = 0; i< max_elements; ++i){
    //for(int i = 0; i< 150; ++i){
    //for(int i = 0; i< 3; ++i){
        data[i] = new float[dim];
        loadin.read((char*)data[i], sizeof(float) * header[2]);

        // Add data to the index
        //std::cout<<"------------------------------------ Adding the " << i << "-th datapoint -----------------------------------" <<std::endl;
        alg_hnsw->addPoint(data[i], i);
        //break;
    
    }
    end = clock();

    std::cout << " Number of data: " << max_elements << " dimentions: " << dim << " building time:" << double(end-start)/CLOCKS_PER_SEC << std::endl;

    //max_elements = 150;
    std::vector<std::vector<int>> clusters;
    int k_means = 128;
    pointsKmeans(alg_hnsw,dim,max_elements,k_means,clusters);

    std::cout << clusters.size() << std::endl;
    for(int i = 0; i<k_means;i++){
        std::cout << clusters[i].size() << std::endl;
    }

    return 0;





    /////////////////////////////////////////////////////////////////////////
    std::cout<<"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" <<std::endl;
    char *currObj = (alg_hnsw->getDataByInternalId(0));
    std::cout << typeid(alg_hnsw->getDataByInternalId(0)).name() << " ";
    std::cout << typeid((alg_hnsw->getDataByInternalId(0))).name() << " ";
    std::cout << typeid(currObj).name() << " ";
    std::cout << typeid(*currObj).name() <<  " " << *currObj << std::endl;

    // unsigned char *a = (unsigned char *) currObj;
    // std::cout << typeid(*a).name() <<  " " << typeid(a).name() << " " << typeid(currObj).name() <<  " " << typeid((unsigned char *) currObj).name() << std::endl;
    // for(int i =0; i<dim; ++i){
    //     std::cout<< int (*a) << " " << a << std::endl;;
    //     a++;
    // }

    /////////////////Function get_linklist
    int ll = 11;
    std::cout << typeid(alg_hnsw->linkLists_[ll]+0*alg_hnsw->size_links_per_element_).name() <<  " " << *(unsigned int *)(alg_hnsw->linkLists_[ll]+0*alg_hnsw->size_links_per_element_) << std::endl;
    std::cout << typeid(alg_hnsw->linkLists_[ll]).name() <<  " " << *alg_hnsw->linkLists_[ll] << std::endl;
    int k = 0;
    std::cout<< alg_hnsw->size_links_per_element_ <<std::endl;
    while(*(unsigned int *)(alg_hnsw->linkLists_[ll]+k) != 0){
        //std::cout <<k << " " <<  *(unsigned int *)(alg_hnsw->linkLists_[ll]+k) <<std::endl;
        ++k;
    }
    //(linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_)
    // int *data_ = (int*)alg_hnsw->get_linklist(11, 1);
    // std::cout << typeid(*data_).name() <<  " " << typeid(data_).name() << std::endl;
    // std::cout << *data_ <<  " " << data_ << std::endl;

    ///////////////Function setListCount
    //*((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    // std::cout << typeid(*((unsigned short int*)(alg_hnsw->linkLists_[ll]))).name() <<  " " << *((unsigned short int*)(alg_hnsw->linkLists_[ll])) << std::endl;
    // for(int i = 0; i<7; ++i){
    //     int *data_ = ( int *) alg_hnsw->linkLists_[ll]+i;
    //     int *data__ = ( int *) (alg_hnsw->linkLists_[ll]+i);
    //     std::cout<<*data_<<" " << *data__ << std::endl;
    // }

    std::cout << typeid(*((unsigned short int*)(alg_hnsw->linkLists_[ll]))).name() <<  " " << *((unsigned short int*)(alg_hnsw->linkLists_[ll])) << std::endl;
    for(int i = 0; i<7; ++i){
        int *data_ = ( int *) alg_hnsw->linkLists_[ll]+i;
        int *data__ = ( int *) (alg_hnsw->linkLists_[ll]+i);
        //std::cout<<*data_<<" " << *data__ << std::endl;
    }


    //return 0;

    // checking Linklist..
    std::cout << "checking Linklist.." << std::endl;
    std::cout << alg_hnsw->cur_element_count << std::endl;
    // for (int i = 0; i < alg_hnsw->cur_element_count; i++) {
    //     if (alg_hnsw->element_levels_[i] > 0)
    //         std::cout << i<< "  "<<alg_hnsw->element_levels_[i] << " " <<  (char *)((char *) alg_hnsw->linkLists_[i])[1] << std::endl;       
    // }

    // std::cout << typeid((char *)((char *) alg_hnsw->linkLists_[11])[1]).name() << " ";
    // std::cout << typeid(((char *) alg_hnsw->linkLists_[11])[1]).name() << " ";
    // std::cout << typeid(alg_hnsw->linkLists_[11][1]).name() << " ";
    // for(int j = 0; j< alg_hnsw->cur_element_count; j++){
    //     std::cout  << " j: "<< j<< " " <<  ((char *) alg_hnsw->linkLists_[11])[j] << std::endl;
    // }
    //std::cout  << " " <<  ((char *) alg_hnsw->linkLists_[11])[29] << std::endl;       
    // std::cout  << " " <<  (char *)((char *) alg_hnsw->linkLists_[11])[1] << std::endl;       

    // for (int i = 0; i < alg_hnsw->cur_element_count; i++) {

    //     std::cout << "datapoint: " << i << ", Level: " << alg_hnsw->element_levels_[i]  << std::endl;
    //     for(int j = 0; j< alg_hnsw->cur_element_count; j++){
    //         std::cout << typeid(( char *)alg_hnsw->linkLists_[i][j]).name() << " ";
    //         if((char *)alg_hnsw->linkLists_[i][j] != nullptr){
    //             std::cout<< (char *)((char *) alg_hnsw->linkLists_[i])[j] <<std::endl;
    //         }
    //     }

    //     // if(alg_hnsw->linkLists_[i] == nullptr){
    //     //     std::cout<< "nullptr" <<std::endl;
    //     // }
    //     // if(alg_hnsw->linkLists_[i] != nullptr ){
    //     //     int j = 0;
    //     //     std::cout<< typeid((char *)alg_hnsw->linkLists_[i][j]).name() <<std::endl;
    //     //     std::cout << alg_hnsw->linkLists_[i][j] << std::endl;
    //     //     // for(int j = 0; j< alg_hnsw->cur_element_count; j++){
    //     //     //     if(alg_hnsw->linkLists_[i][j] != nullptr){
    //     //     //         std::cout<< (char *)((char *) alg_hnsw->linkLists_[i])[j] <<std::endl;
    //     //     //     }
    //     // }

    //     //std::cout << (char *) alg_hnsw->linkLists_[i] << std::endl;
    //     //std::cout << sizeof((char *) alg_hnsw->linkLists_[i]) << " " <<sizeof(((char *) alg_hnsw->linkLists_[i])[0])<<std::endl;
    //     //std::cout << sizeof((char *) alg_hnsw->linkLists_[i])/sizeof(((char *) alg_hnsw->linkLists_[i])[0])<<std::endl;
    //     //  for (int j = 0; j< alg_hnsw->cur_element_count; j++){
    //     //      if ((char *)((char *) alg_hnsw->linkLists_[i])[j])
    //     // }


    //     //if (alg_hnsw->element_levels_[i] > 0)
    //     //std::cout << i<< "  "<<alg_hnsw->element_levels_[i] << " " <<  (char *)((char *) alg_hnsw->linkLists_[i])[1] << std::endl;       
    // }
    

    //return 0 ;


    // // 获取 unordered_map 的开始迭代器
    // auto it = alg_hnsw->label_lookup_.begin();

    // // 遍历 unordered_map，直到达到结束迭代器
    // while (it != alg_hnsw->label_lookup_.end()) {
    //     // 访问当前迭代器指向的键值对
    //     std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;

    //     // 移动到下一个元素
    //     ++it;
    // }

    /////////////////////////////////////// GENERATING RANDOM QUERY//////////////////////////////////
    //Step 1: 
    int query_num = 1000;
    float** query_data = new float* [query_num];


    float correct = 0;
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_int_distribution<int> dis(0, max_elements-1);
    clock_t start_,end_;
    start_ = clock(); 
    for (int i = 0; i < query_num; i++) {
        int query_lable = dis(rng);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data[query_lable], 1);
        hnswlib::labeltype label = result.top().second;
        if (label == query_lable) correct++;
    }
    float recall = correct / query_num;
    end_ = clock();
    std::cout << "Recall: " << recall << "\n";
    std::cout<< "query time: " << double(end_-start_)/CLOCKS_PER_SEC <<std::endl;
    // std::cout << alg_hnsw->label_lookup_.size() << std::endl;
    // std::cout<< alg_hnsw->label_lookup_.begin()->first << "   " << alg_hnsw->label_lookup_.begin()->second <<std::endl;
    

    
    
    /////////////////////////////////////// TESTING //////////////////////////////////
    // std::cout<< alg_hnsw->element_levels_.size() << "  " << sizeof(alg_hnsw->data_level0_memory_)/sizeof(alg_hnsw->data_level0_memory_[0]) << std::endl;

    // // 使用 std::min_element 找出最小值
    // auto min_iter = std::min_element(alg_hnsw->element_levels_.begin(), alg_hnsw->element_levels_.end());
    // int min_value = (*min_iter); // 解引用迭代器以获取实际的值

    // // 使用 std::max_element 找出最大值
    // auto max_iter = std::max_element(alg_hnsw->element_levels_.begin(), alg_hnsw->element_levels_.end());
    // int max_value = (*max_iter); // 解引用迭代器以获取实际的值

    // // 输出最大值和最小值
    // std::cout << "Minimum value in the vector: " << min_value << std::endl;
    // std::cout << "Maximum value in the vector: " << max_value << std::endl;

    // std::cout << sizeof(alg_hnsw->linkLists_[0])/sizeof(alg_hnsw->linkLists_[0][0]) <<std::endl;
    // std::cout << sizeof(alg_hnsw->linkLists_[1])/sizeof(alg_hnsw->linkLists_[1][0]) <<std::endl;
    // std::cout << sizeof(alg_hnsw->linkLists_[2])/sizeof(alg_hnsw->linkLists_[2][0]) <<std::endl;
    // std::cout << sizeof(alg_hnsw->linkLists_[3])/sizeof(alg_hnsw->linkLists_[3][0]) <<std::endl;




    return 0;
}

