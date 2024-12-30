# The Source Code for Shortcut-enabled Hierarchical Graph (SHG) index 

## Introduction

This is a source code for the algorithm described in the paper: Accelerating Approximate Nearest Neighbor Search in
Hierarchical Graphs: Efficient Level Navigation with Shortcuts. The code is developed based on the HNSW Library (https://github.com/nmslib/hnswlib). */pgm* is the learned index for the shortcut. *heds.h* contains the implementation details of our method.

## Compilation 

```bash
mkdir build
cd build
cmake ..
make 
```

## Usage 
*./example_search {algName} {dataset} {k}*

```bash
./example_search heds enron 20
```
Conduct the 20-ANN Search on the dataset Enron. Please unzip the data in */enron* before using it.

## Dataset
In our project, the format of the input file (such as enron.data_new, which is in float data type) is the same as that in existing works. It is a binary file, which is organized as the following format:
```
{Bytes of the data type (int)} {The size of the vectors (int)} {The dimension of the vectors (int)} {All of the binary vector, arranged in turn (float)}
```
For your application, you should also transform your dataset into this binary format, then rename it as [datasetName].data_new and replace the PATH

