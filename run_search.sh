#!/bin/bash

DATA_PATH="/home/data/zgongae/VectorsIndex/datasets/"

SEARCH_DEFAULT="/home/data/zgongae/VectorsIndex/HEDS/build/example_search"


#RESULT_PATH="/home/data/zgongae/VectorsIndex/HEDS/results/"

#run experiments on default synthetic datasets
for data in "enron" "gist" "msong" "uqv" "sift" "msturing" "openai" "deep10m"
#for data in "enron" "gist" "msong" "uqv" "sift" "msturing" "openai" "deep10m" "sift10m"
#for data in "gist" "msong"
do
    for index in "heds"
    do
        for k in 20
        do    
            echo "Benchmark ${index} dataset ${data} -------------------"
            ${SEARCH_DEFAULT} ${index} ${data} ${k}
        done
    done
done