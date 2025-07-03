#!/bin/bash

DATA_PATH="/home/data/zgongae/VectorsIndex/datasets/"

SEARCH_DEFAULT="/home/data/zgongae/VectorsIndex/HEDS/build/example_search"


#RESULT_PATH="/home/data/zgongae/VectorsIndex/HEDS/results/"

#run experiments on default synthetic datasets
#for data in "enron" "gist" "msong" "uqv" "sift" "msturing" "openai" "deep10m"
#for data in "enron" "gist" "msong" "uqv" "sift" "msturing" "openai" "deep10m" "sift10m"
# for data in  "deep100m" "sift100m"
#for data in "msong" "uqv" "gist" "msturing"
for data in "enron"
#for data in "enron" "gist"
do
    for index in "heds" "hnsw"
    #for index in "hnsw"
    do
        for k in 20
        do    
            echo "Benchmark ${index} dataset ${data} -------------------"
            ${SEARCH_DEFAULT} ${index} ${data} ${k}
        done
    done
done
