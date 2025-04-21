#include <iostream>
#include <fstream>
#include <vector>
#include "./UnionFind.h"
#include "SZ3/api/sz.hpp"
// #include <parallel/algorithm>  
// #include <omp.h>
#include <set>
#include <cfloat>
#include <bitset>
#include <filesystem>
#include <map>
// #include <thrust/device_vector.h>
// #include <thrust/sequence.h>
// #include <thrust/find.h>
// #include <thrust/merge.h>
// #include <thrust/sort.h>
// #include <thrust/device_ptr.h>
// #include <thrust/unique.h>

// Key 102: (51, 4)
// Key 153: (12, 4)
// Key 277: (0, 4)
// Key 413: (4, 6)
// Key 554: (63, 4)
// Key 622: (55, 6)
// Key 1050: (15, 4)
// Key 1179: (13, 6)
// Key 1594: (31, 6)
// Key 2085: (48, 4)
// Key 2151: (49, 6)
// Key 2357: (16, 6)
// Key 5462: (3, 7)
// Key 5463: (1, 8)
// Key 5470: (7, 8)
// Key 5494: (19, 8)
// Key 5599: (5, 10)
// Key 6014: (23, 10)
// Key 7543: (17, 10)
// Key 10921: (60, 7)
// Key 10923: (61, 8)
// Key 10925: (52, 8)
// Key 10937: (28, 8)
// Key 10991: (53, 10)
// Key 11197: (20, 10)
// Key 11963: (29, 10)
// Key 16383: (21, 14)

#define MAX_NEIGHBORS 14
#define MAX_V3 6 
#define NUM_LOOKUP 14 // 14 * 14 种 (v1, v2) 组合
#define TILE_SIZE 10
#define BLOCK_SIZE (TILE_SIZE + 2)  // 这里为 10
#define SIZE (NUM_LOOKUP * MAX_V3 * 3 * sizeof(int))  // 计算数组大小
int width_host, height_host, depth_host;
int *adjacency_host;
__device__ int table[64];
// __device__ int lookupTable[NUM_LOOKUP][NUM_LOOKUP][3];
__device__ int lookupTable[NUM_LOOKUP][MAX_V3][3];

int lookupTable_host[NUM_LOOKUP][MAX_V3][3];
int maxNeighbors_host = 14;
int vertex_types[27] = {277, 153, 1050, 1179, 413, 2085, 10921, 554, 10923, 10925, 2357, 10937, 1594, 11963, 11197, 5462, 5463, 5470, 5599, 102, 2151, 622, 10991, 5494, 7543, 6014, 16383};
__constant__ int neighborOffsets[14][3] = {
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0},
    {0,0,1}, {0,0,-1}, {-1,1,0}, {1,-1,0},
    {0,1,1}, {0,-1,-1}, {-1,0,1}, {1,0,-1},
    {-1,1,1}, {1,-1,-1}
};
int neighborOffsets_host[14][3] = {
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0},
    {0,0,1}, {0,0,-1}, {-1,1,0}, {1,-1,0},
    {0,1,1}, {0,-1,-1}, {-1,0,1}, {1,0,-1},
    {-1,1,1}, {1,-1,-1}
};

__device__ const int keys[] = {
    277, 153, 1050, 2085, 102, 554,  // value = 4
    413, 1179, 2357, 1594, 2151, 622, // value = 6
    5462, 10921,  // value = 7
    5463, 5470, 5494, 10937, 10925, 10923,     // value = 8
    5599, 7543, 11197, 6014, 11963, 10991,   // value = 10
    16383  // value = 14
};

__device__ const int lutOffsets[] = {
    0, (1 << 4), (2 << 4), (3 << 4), (4 << 4), (5 << 4),  // value = 4
    6 * (1 << 4), 6 * (1 << 4) + (1 << 6), 6 * (1 << 4) + 2 * (1 << 6),
    6 * (1 << 4) + 3 * (1 << 6), 6 * (1 << 4) + 4 * (1 << 6), 6 * (1 << 4) + 5 * (1 << 6),  // value = 6
    6 * (1 << 4) + 6 * (1 << 6), 6 * (1 << 4) + 6 * (1 << 6) + (1 << 7),  // value = 7
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7), 6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 2 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 3 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 4 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 5 * (1 << 8),  // value = 8
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8),  // value = 10
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 2 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 3 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 4 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 5 * (1 << 10),  // value = 10
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 6 * (1 << 10)  // value = 14
};
const int NUM_KEYS = sizeof(keys) / sizeof(keys[0]);

const int LUT_SIZE = 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 6 * (1<<10) + (1 << 14);

__device__ int LUT_cuda[LUT_SIZE];

int LUT[6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 6 * (1<<10) + (1 << 14)];
// nvcc ExtreM_cuda.cu -o ExtreM_cuda -lzstd
struct OffsetComparator {
    float* d_offset;
    __host__ __device__ OffsetComparator(float* offset) : d_offset(offset) {}

    __host__ __device__ bool operator()(int v, int u) const {
        if (v < 0 || u < 0) return false;
        return d_offset[v] > d_offset[u] || 
               ((abs(d_offset[v] - d_offset[u]) == 0) && (v > u));
    }
};

int initialValue = 0, num_Elements_host;
__device__ float *d_deltaBuffer, *input_data, *decp_data, bound, thr;
__device__ int count_f_max, count_f_min, count_f_saddle, 
                width, height, depth, maxNeighbors, num_Elements, numFaces, num_false_cases, num_false_cases1;
__device__ int *dec_vertex_type, *vertex_type, *vertex_cells, *dec_saddlerank, *saddlerank,
                *dec_saddle1rank, *saddle1rank, *reachable_saddle_for_Max, *dec_reachable_saddle_for_Max,
                *reachable_saddle_for_Min, *dec_reachable_saddle_for_Min,
                *delta_counter, *lowerStars, *dec_lowerStars, 
                *upperStars, *dec_upperStars, *adjacency,
                *AS_M, *DS_M, *dec_AS_M, *dec_DS_M,
                *minimum, *saddles1, *saddles2, *maximum,
                *dec_minimum, *dec_saddles1, *dec_saddles2, *dec_maximum,
                *saddleTriplets, *dec_saddleTriplets, *saddle1Triplets, *dec_saddle1Triplets,
                *tempArray, *dec_tempArray, *tempArrayMin, *dec_tempArrayMin, *largestSaddlesForMax, *dec_largestSaddlesForMax,
                *smallestSaddlesForMin, *dec_smallestSaddlesForMin, *max_index;
__device__ int *or_saddle_max_map, *wrong_neighbors, *wrong_neighbors1, *wrong_neighbors_index, *wrong_rank_max, *wrong_rank_max_index, *wrong_rank_saddle, *wrong_rank_saddle_index, *wrong_rank_max_2, *wrong_rank_max_index_2;
__device__ int *or_saddle_min_map, *wrong_neighbors_ds, *wrong_neighbors_ds_index, *wrong_rank_min, *wrong_rank_min_index, *wrong_rank_min_2, *wrong_rank_min_index_2;
__device__ int *wrong_rank_saddle_join, *wrong_rank_saddle_join_index;
__device__ int nSaddle2 = 0, nSaddle1 = 0, nMin = 0, nMax = 0, dec_nSaddle2 = 0, dec_nSaddle1 = 0, dec_nMin = 0, dec_nMax = 0;
__device__ int number_of_false_cases = 0, wrong_max_counter = 0, wrong_saddle_counter = 0, wrong_saddle_counter_join = 0, wrong_max_counter_2 = 0, globalMin = 0, dec_globalMin = 0;
__device__ int number_of_false_cases1 = 0, wrong_min_counter = 0, wrong_min_counter_2 = 0;
__device__ int *all_max, *all_min, *all_saddle, *updated_vertex;
__device__ int directions[42] = 
{1,0,0,-1,0,0,
0,1,0,0,-1,0,
0,0,1,0,0,-1,
-1,1,0,1,-1,0, 
0,1,1,0,-1,-1,  
-1,0,1,1,0,-1,
-1,1,1,1,-1,-1};

int directions_host[42] = 
{1,0,0,-1,0,0,
0,1,0,0,-1,0,
0,0,1,0,0,-1,
-1,1,0,1,-1,0, 
0,1,1,0,-1,-1,  
-1,0,1,1,0,-1,
-1,1,1,1,-1,-1};


size_t cmpSize = 0;
std::string file_path;
float maxValue, minValue, er, host_bound, host_thre, host_sim;

void getdata(const std::string &filename, float *input_data_host, float *decp_data_host, float *decp_data_copy, 
            const float er, float &bound, int data_size) {
        
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(data_size * sizeof(float))) {
        std::cerr << "File size does not match expected data size." << std::endl;
        return;
    }

    std::vector<float> h_buffer(data_size);
    file.read(reinterpret_cast<char *>(h_buffer.data()), size);
    if (!file) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }

    cudaMemcpy(input_data_host, h_buffer.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(input_data, &input_data_host, sizeof(float*));
    
    SZ3::Config conf(static_cast<size_t>(depth_host),
                 static_cast<size_t>(height_host),
                 static_cast<size_t>(width_host));

    conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
    conf.errorBoundMode = SZ3::EB_REL;
    conf.relErrorBound = er; 

    
    
    char *compressedData = SZ_compress(conf, h_buffer.data(), cmpSize);

    
    float *d_buffer = new float[data_size];
    SZ_decompress(conf, compressedData, cmpSize, d_buffer);
    
    
    cudaMemcpy(decp_data_host, d_buffer, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(decp_data_copy, d_buffer, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(decp_data, &decp_data_host, sizeof(float*));
    delete[] compressedData;

    
    float minValue = *std::min_element(h_buffer.begin(), h_buffer.end());
    float maxValue = *std::max_element(h_buffer.begin(), h_buffer.end());
    host_thre = (maxValue - minValue) * host_sim;
    bound = (maxValue - minValue) * er;
    
    std::cout << "Data read, compressed, and decompressed successfully." << std::endl;
}

int calculateNeighborIndex(int nx, int ny, int nz){
    for(int i = 0; i<maxNeighbors_host; i++){
        if(nx == neighborOffsets_host[i][0] && ny == neighborOffsets_host[i][1] && nz == neighborOffsets_host[i][2]) return i;
    }
    return -1;
}

void loadLUTToGPU() {
    // 1. 在 CPU 上分配内存并读取 LUT.bin
    int* h_LUT = new int[LUT_SIZE];
    std::ifstream file("LUT.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Error: Failed to open LUT.bin" << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(h_LUT), LUT_SIZE * sizeof(int));
    file.close();

    // 2. 将 CPU 的 LUT 复制到 __device__ 变量
    cudaMemcpyToSymbol(LUT_cuda, h_LUT, LUT_SIZE * sizeof(int));

    // 3. 释放 CPU 内存
    delete[] h_LUT;
}


int calculateLUT(const float* heightMap, int i, int tableSize, int startIndex, int type=0) {
    
    int vertexId = i;
    int y = (i / (width_host)) % height_host; // Get the x coordinate
    int x = i % width_host; // Get the y coordinate
    int z = (i / (width_host * height_host)) % depth_host;
    
    for(int config = 0; config<tableSize;config++){
        int neighbor_size = 0;
        std::bitset<14> binary(config); 
        
        std::vector<std::vector<int>> *upperComponents = nullptr;
        std::vector<std::vector<int>> *lowerComponents = nullptr;

        std::vector<std::vector<int>> localUpperComponents;
        std::vector<std::vector<int>> localLowerComponents;
        if(upperComponents == nullptr) {
            upperComponents = &localUpperComponents;
        }
        if(lowerComponents == nullptr) {
            lowerComponents = &localLowerComponents;
        }
        
        std::vector<int > lowerStar, upperStar;
        
        int neighbor[MAX_NEIGHBORS]= {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        for (int d =0;d<maxNeighbors_host;d++) {
            // if(adjacency_host[vertexId*maxNeighbors_host+d]==-1) continue;
            // int r = adjacency_host[i*maxNeighbors_host+d];
            int dx = directions_host[3 * d];
            int dy = directions_host[3 * d + 1];
            int dz = directions_host[3 * d + 2];

            int nx = x + dx;
            int ny = y + dy;
            int nz = z + dz;
            if(nx < 0 || nx >= width_host || ny < 0 || ny >= height_host || nz < 0 || nz >= depth_host) continue;
            int r = x + dx + (y + dy + (z + dz) * height_host) * width_host;
            neighbor[neighbor_size] = r;
            if (binary[neighbor_size] == 0) {
                lowerStar.emplace_back(r);
            } 
            else{
                upperStar.emplace_back(r);
            }
            neighbor_size++;
        }
        // if(neighbor_size == 8 && config == 44){
        //     // printf("unmatched here !!!%d: %d %d %d %d\n", g_idx, result, LUT_result, neighbor_size, decimal_value);
        //     std::cout<<binary<<", "<< i<<std::endl;
        //     printf("\n");
        //     for(auto i:upperStar) printf("%d ",i);
        //     printf("\n");
        // }
        std::vector<UnionFind> lowerSeeds(lowerStar.size());
        std::vector<UnionFind> upperSeeds(upperStar.size());
        std::vector<UnionFind*> lowerList(lowerStar.size());
        std::vector<UnionFind*> upperList(upperStar.size());

        std::vector<int> lowerNeighbors, upperNeighbors;
        upperNeighbors = upperStar;
        lowerNeighbors = lowerStar;
    
        for(int i = 0; i < (int)lowerList.size(); i++)
            lowerList[i] = &(lowerSeeds[i]);
        for(int i = 0; i < (int)upperList.size(); i++)
            upperList[i] = &(upperSeeds[i]);
        
        int neighbor_idx = 0;
        for(int t = 0; t < 14; t++){
            int dx = directions_host[ t * 3 + 0];
            int dy = directions_host[ t * 3 + 1];
            int dz = directions_host[ t * 3 + 2];

            int nx = x + dx;
            int ny = y + dy;
            int nz = z + dz;
            int v2 = nx + (ny + (nz) * height_host) * width_host;
            
            if(nx >= width_host || ny >= height_host || nz >= depth_host || nx < 0 ||
                ny < 0 || nz < 0 || v2 >= num_Elements_host || v2 < 0) continue;
            bool const lower0 = binary[neighbor_idx] == 0;
            neighbor_idx++;
            for(int j = 0; j < MAX_V3; j++){
                int dx1 = lookupTable_host[t][j][0];
                int dy1 = lookupTable_host[t][j][1];
                int dz1 = lookupTable_host[t][j][2];
                // int cellId = vertex_cells[i * 42 + t];
                if(dx1 == -2 || dy1 == -2 || dz1 == -2) break;
                int nx1 = x + dx1;
                int ny1 = y + dy1;
                int nz1 = z + dz1;
                int v3 = nx1 + (ny1 + (nz1) * height_host) * width_host;
                
                if(nx1 >= width_host || ny1 >= height_host || nz1 >= depth_host || nx1 < 0 ||
                    ny1 < 0 || nz1 < 0 || v3 >= num_Elements_host || v3 < 0 || v3 < v2) continue;
                int idx;
                for(int k = 0; k<neighbor_size; k++){
                    if(v3 == neighbor[k]){
                        idx = k;
                        break;
                    }
                }
                bool const lower1 = binary[idx] == 0;
                std::vector<int> *neighbors = &lowerNeighbors;
                std::vector<UnionFind *> *seeds = &lowerList;

                if(!lower0) {
                    neighbors = &upperNeighbors;
                    seeds = &upperList;
                }

                if(lower0 == lower1) {
                    // connect their union-find sets!
                    int lowerId0 = -1, lowerId1 = -1;
                    for(int l = 0; l < (int)neighbors->size(); l++) {
                        if((*neighbors)[l] == v2) {
                            
                            lowerId0 = l;
                        }
                        if((*neighbors)[l] == v3) {
                            lowerId1 = l;
                        }
                    }
                    if((lowerId0 != -1) && (lowerId1 != -1)) {
                        // if(vertexId == 0) cout<<lowerId0<<", "<<lowerId1<<endl;
                        (*seeds)[lowerId0] = UnionFind::makeUnion(
                            (*seeds)[lowerId0], (*seeds)[lowerId1]);
                            (*seeds)[lowerId1] = (*seeds)[lowerId0];
                        }
                }
            }
        }


       

        
        // update the UF if necessary
        for(int i = 0; i < (int)lowerList.size(); i++)
            lowerList[i] = lowerList[i]->find();
        for(int i = 0; i < (int)upperList.size(); i++)
        {
            upperList[i] = upperList[i]->find();
        }
            

        std::unordered_map<UnionFind *, std::vector<int>>::iterator it;
        std::unordered_map<UnionFind *, std::vector<int>>
            upperComponentId{};
        std::unordered_map<UnionFind *, std::vector<int>>
            lowerComponentId{};
        // We retrieve the lower and upper components if we want them
        for(int i = 0; i < (int)upperNeighbors.size(); i++) {
            
            it = upperComponentId.find(upperList[i]);
            if(it != upperComponentId.end()) {
                upperComponentId[upperList[i]].push_back(upperNeighbors[i]);
            } else {
            upperComponentId[upperList[i]]
                = std::vector<int>(1, upperNeighbors[i]);
            }
        }

        
        for(auto elt : upperComponentId) {
            upperComponents->push_back(std::vector<int>());
            
            for(int i = 0; i < (int)elt.second.size(); i++) {
                upperComponents->back().push_back(elt.second.at(i));
            }
        }
    
        

        for(int i = 0; i < (int)lowerNeighbors.size(); i++) {
            it = lowerComponentId.find(lowerList[i]);
            if(it != lowerComponentId.end()) {
                lowerComponentId[lowerList[i]].push_back(lowerNeighbors[i]);
            } else {
            lowerComponentId[lowerList[i]]
                = std::vector<int>(1, lowerNeighbors[i]);
            }
        }

        for(auto elt : lowerComponentId) {
            lowerComponents->push_back(std::vector<int>());
            for(int i = 0; i < (int)elt.second.size(); i++) {
                lowerComponents->back().push_back(elt.second.at(i));
            }
        }
        
        int lowerComponentNumber = lowerComponents->size();
        int upperComponentNumber = upperComponents->size();

        // minimum: 0 1-saddle: 1, 2-saddle: 2, regular: 5 maximum: 4;
        if(lowerComponentNumber == 0 && upperComponentNumber == 1) LUT[startIndex + config]=0;
        else if(lowerComponentNumber == 1 && upperComponentNumber == 0) LUT[startIndex + config]=4;
        else if(lowerComponentNumber == 1 && upperComponentNumber == 1) LUT[startIndex + config]=5;
        else if(lowerComponentNumber > 1 && upperComponentNumber > 1) LUT[startIndex + config]=3;
        else if(lowerComponentNumber > 1) LUT[startIndex + config]=1;
        else if(upperComponentNumber > 1) LUT[startIndex + config]=2;
    }      
    return 0;
}

__device__ void getVerticesFromTriangleID_2d(
        int triangleID,
        int &v1, 
        int &v2, 
        int &v3) {


        int baseID, min_x, min_y, min_z;
        if (triangleID < 2 * (width - 1) * (height - 1)) {
            baseID = triangleID / 2;
            min_z = 0;
            min_y = (baseID % ((width - 1) * (height - 1))) / (width - 1);
            min_x = baseID % (width - 1);

            // 三角形类型（上下三角形）
            if (triangleID % 2 == 0) {
                v1 = min_x + min_y * width + min_z * width * height;
                v2 = (min_x + 1) + min_y * width + min_z * width * height;
                v3 = min_x + (min_y + 1) * width + min_z * width * height;
            } else {
                v1 = (min_x + 1) + (min_y + 1) * width + min_z * width * height;
                v2 = (min_x) + (min_y + 1)* width + min_z * width * height;
                v3 = min_x + 1 + (min_y) * width + min_z * width * height;
            }

            if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
            if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
            if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
            return;
            
        }

        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        
        return; 
}

__device__ void getVerticesFromTriangleID_3d(
    int triangleID,
    int &v1, 
    int &v2, 
    int &v3) {

    int baseID, min_x, min_y, min_z;
    int starID = triangleID;
    
    if (triangleID < 2 * depth * (width - 1) * (height - 1)) {
        baseID = triangleID / 2;
        min_z = baseID / ((width - 1) * (height - 1));
        min_y = (baseID % ((width - 1) * (height - 1))) / (width - 1);
        min_x = baseID % (width - 1);

        if (triangleID % 2 == 0) {
            v1 = min_x + min_y * width + min_z * width * height;
            v2 = (min_x + 1) + min_y * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + min_z * width * height;
        } else {
            v1 = (min_x + 1) + (min_y + 1) * width + min_z * width * height;
            v2 = (min_x + 1) + min_y * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + min_z * width * height;
        }

        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        
        return;
        
    }

    triangleID -= 2 * depth * (width - 1) * (height - 1);
    if (triangleID < 2 * width * (height - 1) * (depth - 1)) {
        baseID = triangleID / 2;
        min_x = baseID / ((height - 1) * (depth - 1));
        min_y = (baseID % ((height - 1) * (depth - 1))) / (depth - 1);
        min_z = baseID % (depth - 1);
        
        if (triangleID % 2 == 0) {
            v1 = min_x + min_y * width + min_z * width * height;
            v2 = min_x + (min_y + 1) * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + (min_z + 1) * width * height;
        } else {
            v1 = min_x + (min_y) * width + (min_z) * width * height;
            v2 = min_x + (min_y) * width + (min_z + 1) * width * height;
            v3 = min_x + (min_y+1) * width + (min_z + 1) * width * height;
        }
        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        // if(starID == 2945) printf("type2: %d %d %d %d %d\n", v1, v2, v3, starID, triangleID);
        return;
    }

    // XZ 平面的三角形
    triangleID -= 2 * width * (height - 1) * (depth - 1);
    if (triangleID < 2 * height * (width - 1) * (depth - 1)) {
        baseID = triangleID / 2;
        min_y = baseID / ((width - 1) * (depth - 1));
        min_z = (baseID % ((width - 1) * (depth - 1))) / (width - 1);
        min_x = baseID % (width - 1);

        if (triangleID % 2 == 0) {
            v1 = min_x + min_y * width + min_z * width * height;
            v2 = (min_x + 1) + min_y * width + min_z * width * height;
            v3 = min_x + min_y * width + (min_z + 1) * width * height;
        } else {
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x) + min_y * width + (min_z + 1) * width * height;
            v3 = (min_x+1) + min_y * width + (min_z + 1) * width * height;
        }
        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        // if(starID == 2945) printf("type3: %d %d %d\n", v1, v2, v3, starID);
        return;
    }

    // 对角线方向的三角形
    triangleID -= 2 * height * (width - 1) * (depth - 1);
    baseID = triangleID / 6;
    min_z = baseID / ((width - 1) * (height - 1));
    min_y = (baseID % ((width - 1) * (height - 1))) / (width - 1);
    min_x = baseID % (width - 1);
    // int y2 = triangleID / ((width - 1) * (depth - 1));
    // int z2 = (triangleID % ((width - 1) * (depth - 1))) / (depth - 1);
    // int x2 = triangleID % (depth - 1);

    int subTriangleType = triangleID % 6;
    
    switch (subTriangleType) {
        
        case 0:
            v1 = min_x + (min_y) * width + (min_z) * width * height;
            v2 = min_x+1 + min_y * width + min_z * width * height;
            v3 = (min_x ) + (min_y +1)* width + (min_z+1) * width * height;
            break;
        case 1:
            v1 = (min_x +1) + (min_y ) * width + (min_z) * width * height;
            v2 = min_x + (min_y + 1) * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + (min_z+1)* width * height;
            break;
        case 2:
            v1 = min_x + 1 + min_y * width + min_z * width * height;
            v2 = (min_x + 1) + (min_y + 1) * width + min_z * width * height;
            v3 = (min_x) + (min_y +1)* width + (min_z +1) * width * height;
            break;
        case 3:
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x) + min_y * width + ( min_z+1) * width * height;
            v3 = min_x + (min_y + 1) * width + (min_z + 1) * width * height;
            break;
        case 4:
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x + 1) + (min_y) * width +( min_z +1 ) * width * height;
            v3 = (min_x) + (min_y + 1) * width + (min_z + 1)* width * height;
            break;
        case 5:
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x) + (min_y + 1) * width + (min_z + 1) * width * height;
            v3 = (min_x + 1) + (min_y + 1) * width +( min_z+1) * width * height;
            break;
    }

    // if(starID == 2945) printf("type4: %d %d %d %d %d %d\n", v1, v2, v3, starID, triangleID, subTriangleType);
    if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
    if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
    if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
    
    return;
    
}



__global__ void computeAdjacency() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
    
    int x = i % width; // Get the y coordinate
    int y = (i / (width)) % height; // Get the x coordinate
    int z = (i / (width * height)) % depth;
    int neighborIdx = 0;
    // int binary[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int d = 0; d < maxNeighbors; d++) {
        
        int dirX = directions[d * 3];     
        int dirY = directions[d * 3 + 1]; 
        int dirZ = directions[d * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            adjacency[i * maxNeighbors + neighborIdx] = r;
            neighborIdx++;
            // binary[13 - d] = 1;
        }
        
    }

    // int decimal_value = 0;
    // for (int i = 0; i < maxNeighbors; i++) {
    //     decimal_value = (decimal_value << 1) | binary[i]; // 左移并加上当前位
    // }
    
    // Fill the remaining slots with -1 or another placeholder value
    for (int j = neighborIdx; j < maxNeighbors; ++j) {
        adjacency[i * maxNeighbors + j] = -1;
    }
    
    // if((x == 0 || y == 0 || z == 0 || x == width - 1 || y == height - 1 || z == depth - 1) && (x != width - 2 && y != height - 2 && z != depth - 2) ){
    //     printf("%d %d %d %d %d \n", i, neighborIdx, x, y, z);
    //     // for(int j = 0; j<maxNeighbors; j++) printf("%d ", adjacency[i * maxNeighbors + j]);
        
    // }
    // printf("%d %d: %d\n", decimal_value,i,neighborIdx );
}

void computeAdjacency_host() {
    
    std::map<int, std::pair<int, int>> myMap;
    for(int i = 0; i<num_Elements_host;i++){
        int x = i % width_host; // Get the y coordinate
        int y = (i / (width_host)) % height_host; // Get the x coordinate
        int z = (i / (width_host * height_host)) % depth_host;
        int neighborIdx = 0;
        int binary[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int d = 0; d < maxNeighbors_host; d++) {
            
            int dirX = directions_host[d * 3];     
            int dirY = directions_host[d * 3 + 1]; 
            int dirZ = directions_host[d * 3 + 2]; 
            int newX = x + dirX;
            int newY = y + dirY;
            int newZ = z + dirZ;
            int r = newX + newY * width_host + newZ* (height_host * width_host); // Calculate the index of the adjacent vertex
            
            if (newX >= 0 && newX < width_host && newY >= 0 && newY < height_host && r >= 0 && r < width_host*height_host*depth_host && newZ<depth_host && newZ>=0) {
                
                neighborIdx++;
                binary[13 - d] = 1;
            }
            
        }

        int decimal_value = 0;
        for (int j = 0; j < maxNeighbors_host; j++) {
            decimal_value = (decimal_value << 1) | binary[j]; // 左移并加上当前位
        }
        if(myMap.count(decimal_value) == 0){
            
            myMap[decimal_value] = std::make_pair(i, neighborIdx);
        }
    }
    // int num_types = 0;
    // for (const auto& pair : myMap) {
    //     std::cout << "Key " << pair.first << ": (" << pair.second.first << ", " << pair.second.second << ")\n";
    //     int n = pair.second.second;
    //     num_types+=1<<n;
    // }
    std::vector<std::pair<int, std::pair<int, int>>> vec(myMap.begin(), myMap.end());

    // 2️⃣ 使用 `std::sort()` 排序
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        if (a.second.second != b.second.second)
            return a.second.second < b.second.second;  // 先按 second 排序
        return a.second.first < b.second.first;  // second 相同时，按 first 排序
    });

    // 3️⃣ 输出排序后的结果
    for (const auto& [key, value] : vec) {
        std::cout << "Key: " << key << ", Value: (" << value.first << ", " << value.second << ")\n";
    }
    // std::cout <<num_types <<std::endl;

    
}

__device__ int getTriangleID_internal(int v1, int v2, int v3, int width, int height, int depth, int direction = 0) {
    int id = -1;
    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    
    
    int min_x = min(v1%width, min(v2%width, v3%width));
    int min_y = min((v1/width)%height, min((v2/width)%height, (v3/width)%height));
    int min_z = min((v1 / (width * height)) % depth, min((v2 / (width * height)) % depth, (v3 / (width * height)) % depth));
    
    // xy palne
    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    if(direction == 0){

        int cell_id = min_y * (width - 1) + min_x;
        if(sortedV3_x == sortedV1_x && sortedV1_x == min_x && sortedV3_y == sortedV2_y && sortedV2_y == min_y){
            return cell_id * 2 + 2 * min_z * (width - 1) * (height - 1);
        }

        else{
            return cell_id * 2 + 1 + 2 * min_z * (width - 1) * (height - 1);
        }
    }
    // yz plane
    else if(direction == 1){
        int cell_id = min_y * (depth - 1) + min_z;
        if(sortedV2_y == sortedV1_y && sortedV1_y == min_y + 1 && sortedV3_z == sortedV2_z && sortedV2_z == min_z){
            return cell_id * 2 + 2 * depth  * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
        else{
            return cell_id * 2 + 1 + 2 * depth * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
    }
    //  xz plane
    else if(direction == 2){
        int cell_id = min_z * (width - 1) + min_x;
        if(sortedV2_z == sortedV3_z && sortedV2_z == min_z && sortedV1_x == sortedV3_x && sortedV1_x == min_x){
            id = cell_id * 2 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
            return id;
        }
        else{
            return cell_id * 2 + 1 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
        }

    }

    else if(direction == 3){
        int cell_id = min_z * (width - 1) * (height - 1) + min_y * (width - 1) + min_x;
        int current_id = -1;
        if(sortedV3_x == min_x && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1&& sortedV2_y == min_y && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 0;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y + 1 && sortedV1_z == min_z+1 
        ) {current_id = 1;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 2;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 3;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 4;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x + 1 && sortedV1_y== min_y + 1 && sortedV1_z == min_z +1 
        ) {current_id = 5;}
        id = cell_id * 6 + current_id + 2 * depth * (width - 1) * (height - 1) + 2 * width * (height - 1) * (depth - 1) +  2 * height * (width - 1) * (depth - 1);
       
        return id;
    }
    return -1;
    

}

__device__ int find(int* parent, int x) {
    int root = x;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (x != root) {
        int old = parent[x];
        parent[x] = root;  // 一次性路径压缩
        x = old;
    }
    return root;
}

__device__ void unionSets(int *parent, int x, int y) {
    int rootX = find(parent, x);
    int rootY = find(parent, y);
    if (rootX != rootY) {
        parent[rootY] = rootX; // 连接两个集合
    }
}

__device__ void union_sets(int *parent, int *rank, int x, int y) {
    int rootX = find(parent, x);
    int rootY = find(parent, y);
    if (rootX != rootY) {
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else {
            parent[rootY] = rootX;
            rank[rootX] += 1;
        }
    }
}


__device__ bool areVectorsDifferent(int i, int type=0){
    int *stars = lowerStars;
    int *dec_stars = dec_lowerStars;

    if(type == 1){
        stars = upperStars;
        dec_stars = dec_upperStars;
    }
    // if(i == 782) printf("%d %d\n", stars[(maxNeighbors+1) * i + maxNeighbors], dec_stars[(maxNeighbors+1) * i + maxNeighbors]);
    if(stars[(maxNeighbors+1) * i + maxNeighbors] != dec_stars[(maxNeighbors+1) * i + maxNeighbors]){
        return true;
    }

    int starNumber = stars[(maxNeighbors+1) * i + maxNeighbors];
    for (size_t j = 0; j < starNumber; j++) {
        // if(i == 782) printf("%d %d\n", stars[(maxNeighbors+1) * i +j], dec_stars[(maxNeighbors+1) * i +j]);
        if (stars[(maxNeighbors+1) * i + j] != dec_stars[(maxNeighbors+1) * i + j]) {
            return true; 
        }
    }
    return false;
}

__device__ bool areVectorsDifferent_local(int i, int *dec_stars, int lowerCount){
    int *stars = lowerStars + (maxNeighbors+1) * i;
    if(stars[maxNeighbors] != lowerCount){
        return true;
    }
    for (size_t j = 0; j < lowerCount; j++) {
        // if(i == 782) printf("%d %d\n", stars[(maxNeighbors+1) * i +j], dec_stars[(maxNeighbors+1) * i +j]);
        if (stars[j] != dec_stars[j]) {
            return true; 
        }
    }
    return false;
}

__device__ int binarySearchLUT(const int* keys, int numKeys, int target) {
    int left = 0, right = numKeys - 1;
    for(int i = 0; i< numKeys; i++){
        if(target == keys[i]) return i;
    }
    return -1;
    // while (left <= right) {
    //     int mid = left + (right - left) / 2;
    //     if (keys[mid] == target) return mid;
    //     if (keys[mid] < target) left = mid + 1;
    //     else right = mid - 1;
    // }
    // return -1; // Not found
}



__global__ void cloops_kernel_CUDA(int type = 0, int local = 1) {
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= num_Elements ) return;
    // if(updated_vertex[i] != 0 && local == 1) return;

    float* heightMap = input_data;
    int* results = vertex_type;
    int* lowerStars_t = lowerStars;
    int* upperStars_t = upperStars;

    if(type == 1){
        heightMap = decp_data;
        results = dec_vertex_type;
        lowerStars_t = dec_lowerStars;
        upperStars_t = dec_upperStars;
    }

    __shared__ float smem[TILE_SIZE + 2][TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x + 1;  // 1-based index in shared memory (考虑Halo)
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    int gx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int gy = blockIdx.y * TILE_SIZE+ threadIdx.y;
    int gz = blockIdx.z * TILE_SIZE + threadIdx.z;
    
    // 读取当前 voxel
    if (gx < width && gy < height && gz < depth) {
        smem[tx][ty][tz] = heightMap[gz * width * height + gy * width + gx];
    }

    
    if( threadIdx.x == 0 || threadIdx.x == TILE_SIZE - 1 ||
        threadIdx.y == 0 || threadIdx.y == TILE_SIZE - 1  ||
        threadIdx.z == 0 || threadIdx.z == TILE_SIZE - 1 ){
        for (int i = 0; i < 14; i++) {
            int dx = neighborOffsets[i][0];
            int dy = neighborOffsets[i][1];
            int dz = neighborOffsets[i][2];

            int nx = gx + dx;
            int ny = gy + dy;
            int nz = gz + dz;

            int ntx = tx + dx;
            int nty = ty + dy;
            int ntz = tz + dz;

            // 只有 block 边界的线程负责加载 Halo Cells
            if(nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz <depth){
                smem[ntx][nty][ntz] = heightMap[nz * height * width + ny * width + nx];
            }
        }
    }


    __syncthreads();  // 确保所有 shared memory 都加载完毕
    

    if (!(gx < width && gy < height && gz < depth))  return;
    
    int g_idx = gz * width * height + gy * width + gx;
    
        
    float currentHeight = smem[tx][ty][tz];
    int vertexId = g_idx;

    int lowerCount = 0, upperCount = 0;
    int lowerStar[MAX_NEIGHBORS] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int upperStar[MAX_NEIGHBORS]= {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    for (int d = 0; d < maxNeighbors; d++) {
        int r = adjacency[vertexId * maxNeighbors + d];
        if (r == -1) break;
        int nx = r % width;
        int ny = (r / width) % height;
        int nz = (r / (width * height)) % depth;
        int smem_x = tx + (nx - gx);
        int smem_y = ty + (ny - gy);
        int smem_z = tz + (nz - gz);
        
        // float neighbor_value = heightMap[r];
        // if(smem_z * (BLOCK_SIZE * BLOCK_SIZE) + smem_y * BLOCK_SIZE + smem_x >= BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE){
        //     printf("%d %d %d %d %d %d %d %d %d\n", nx, ny, nz, gx, gy, gz, smem_x, smem_y, smem_z);
        // }
        float neighbor_value = smem[smem_x][smem_y][smem_z];

        if (neighbor_value < currentHeight || (neighbor_value == currentHeight && r < g_idx)) {
            lowerStars_t[vertexId*(maxNeighbors+1) + lowerCount] = r;
            lowerStar[lowerCount] = r;
            lowerCount++;
            
        } else if (neighbor_value > currentHeight || (neighbor_value == currentHeight && r > g_idx)) {
            upperStars_t[vertexId*(maxNeighbors+1) + upperCount] = r;
            upperStar[upperCount] = r;
            upperCount++;
            
        }
    }

    lowerStars_t[vertexId*(maxNeighbors+1) + maxNeighbors] = lowerCount;
    upperStars_t[vertexId*(maxNeighbors+1) + maxNeighbors] = upperCount;

    
    int lowerParent[MAX_NEIGHBORS]= {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int upperParent[MAX_NEIGHBORS]= {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    
    int lowerRank[MAX_NEIGHBORS]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int upperRank[MAX_NEIGHBORS]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < lowerCount; j++) {
        lowerParent[j] = j;
        lowerRank[j] = 0;
    }
    for (int j = 0; j < upperCount; j++){
        upperParent[j] = j;
        upperRank[j] = 0;
    }
    
    int x = vertexId % width;
    int y = (vertexId / width ) % height;
    int z = (vertexId / (width *  height)) % depth;
    for(int t = 0; t < 14; t++){
        int dx = directions[ t * 3 + 0];
        int dy = directions[ t * 3 + 1];
        int dz = directions[ t * 3 + 2];

        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;
        int v2 = nx + (ny + (nz) * height) * width;
        
        if(nx >= width || ny >= height || nz >= depth || nx < 0 ||
            ny < 0 || nz < 0 || v2 >= num_Elements || v2 < 0) continue;

        for(int j = 0; j < 14; j++){
            int dx1 = lookupTable[t][j][0];
            int dy1 = lookupTable[t][j][1];
            int dz1 = lookupTable[t][j][2];
            // int cellId = vertex_cells[i * 42 + t];
            if(dx1 == -2 || dy1 == -2 || dz1 == -2) continue;
            int nx1 = x + dx1;
            int ny1 = y + dy1;
            int nz1 = z + dz1;
            int v3 = nx1 + (ny1 + (nz1) * height) * width;
            if(nx1 >= width || ny1 >= height || nz1 >= depth || nx1 < 0 ||
                ny1 < 0 || nz1 < 0 || v3 >= num_Elements || v3 < 0 || v3 == g_idx) continue;
            int vertices[3] = {vertexId, v2, v3};
            
            for(int j = 0; j<3; j++){
                int neighborId0 = vertices[j];
                if(neighborId0 == vertexId) continue;
                int nx = neighborId0 % width;
                int ny = (neighborId0 / width) % height;
                int nz = (neighborId0 / (width * height)) % depth;
                int smem_x = tx + (nx - gx);
                int smem_y = ty + (ny - gy);
                int smem_z = tz + (nz - gz);
                float neighbor0_value = smem[smem_x][smem_y][smem_z];;
                // float neighbor0_value = heightMap[neighborId0];
                bool const lower0 = neighbor0_value < currentHeight || (neighbor0_value == currentHeight and neighborId0<vertexId);
                for(int k = j + 1; k < 3; k++) {
                    int neighborId1 = vertices[k];
                    if((neighborId1 != neighborId0) && (neighborId1 != g_idx)) {
                        int nx = neighborId1 % width;
                        int ny = (neighborId1 / width)%height;
                        int nz = (neighborId1 / (width * height)) % depth;
                        int smem_x = tx + (nx - gx);
                        int smem_y = ty + (ny - gy);
                        int smem_z = tz + (nz - gz);
                        float neighbor1_value = smem[smem_x][smem_y][smem_z];;
                        // float neighbor1_value = heightMap[neighborId1];
                        bool const lower1 = neighbor1_value < currentHeight || (neighbor1_value == currentHeight and neighborId1<vertexId);
                        int *neighbors = lowerStar;
                        int *seeds = lowerParent;
                        int *rank = lowerRank;

                        if(!lower0) {
                            neighbors = upperStar;
                            seeds = upperParent;
                            rank = upperRank;
                        }

                        if(lower0 == lower1) {
                            // connect their union-find sets!
                            int lowerId0 = -1, lowerId1 = -1;
                            for(int l = 0; l < maxNeighbors; l++) {
                                if(neighbors[l] == -1) break;
                                if(neighbors[l] == neighborId0) {
                                    lowerId0 = l;
                                }
                                if(neighbors[l] == neighborId1) {
                                    lowerId1 = l;
                                }
                            }
                            if((lowerId0 != -1) && (lowerId1 != -1)) {
                                
                                union_sets(seeds, rank,lowerId0, lowerId1);
                            }
                        }
                    }
                }
            }
        }

    }


    int lowerComponentCount = 0;
    int upperComponentCount = 0;
    for (int j = 0; j < lowerCount; j++) {
        if (find(lowerParent, j) == j) lowerComponentCount++;
    }
    for (int j = 0; j < upperCount; j++) {
        if (find(upperParent, j) == j) upperComponentCount++;
    }

    int result = 5;  

    if (lowerComponentCount == 0 && upperComponentCount == 1) result = 0; 
    else if (lowerComponentCount == 1 && upperComponentCount == 0) result = 4; 
    else if (lowerComponentCount == 1 && upperComponentCount == 1) result = 5; 
    else if (lowerComponentCount > 1 && upperComponentCount > 1) result = 3; 
    else if (lowerComponentCount > 1) result = 1;
    else if (upperComponentCount > 1) result = 2;
    
    results[g_idx] = result;
}




__device__ int getTriangleID_2d(int v1, int v2, int v3, int width, int height, int depth){

    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    

    if(sortedV1_z == sortedV2_z && sortedV2_z == sortedV3_z){
        int id = getTriangleID_internal(sortedV1, sortedV2, sortedV3, width, height, depth) ;
        return id;
    }
    // yz plane
    if(sortedV1_x == sortedV2_x && sortedV2_x == sortedV3_x){
        int id = getTriangleID_internal(sortedV1, sortedV2, sortedV3, width, height, depth, 1) ;
        return id;
    }

    
    
    if(sortedV1_y == sortedV2_y && sortedV2_y == sortedV3_y){
        int id = getTriangleID_internal(sortedV1, sortedV2, sortedV3, width, height, depth, 2);
        return id;
    }
    
    else{
        int id = getTriangleID_internal(sortedV1, sortedV2, sortedV3, width, height, depth, 3);
        return id;
    }

    return -1;
}

__device__ int getTriangleID_3d_internel(int v1, int v2, int v3, int width, int height, int depth, int direction = 0) {
    int id = -1;
    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    
    
    int min_x = min(v1%width, min(v2%width, v3%width));
    int min_y = min((v1/width)%height, min((v2/width)%height, (v3/width)%height));
    int min_z = min((v1 / (width * height)) % depth, min((v2 / (width * height)) % depth, (v3 / (width * height)) % depth));
    
    // xy palne
    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    if(direction == 0){

        int cell_id = min_y * (width - 1) + min_x;
        if(sortedV3_x == sortedV1_x && sortedV1_x == min_x && sortedV3_y == sortedV2_y && sortedV2_y == min_y)
        {
            
            return cell_id * 2 + 2 * min_z * (width - 1) * (height - 1);
        }

        else
        {
            return cell_id * 2 + 1 + 2 * min_z * (width - 1) * (height - 1);
        }
    }
    // yz plane
    else if(direction == 1){
        int cell_id = min_y * (depth - 1) + min_z;
        if(sortedV2_y == sortedV1_y && sortedV1_y == min_y + 1 && sortedV3_z == sortedV2_z && sortedV2_z == min_z)
        {
            
            return cell_id * 2 + 2 * depth  * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
        else
        {
            // printf("%d\n",cell_id * 2 + 1 + 2 * depth * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1));
            
            
            return cell_id * 2 + 1 + 2 * depth * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
    }
    //  xz plane
    // if(sortedV1_x == 9 && sortedV1_y == 9 && sortedV1_z == 9) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
    else if(direction == 2){
        int cell_id = min_z * (width - 1) + min_x;
        // if(sortedV1_x == 9 && sortedV1_y == 9 && sortedV1_z == 9) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
        if(sortedV2_z == sortedV3_z && sortedV2_z == min_z && sortedV1_x == sortedV3_x && sortedV1_x == min_x)
        {
            id = cell_id * 2 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
            // if(id == 4858) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
            
            return id;
        }
        else
        {
            
            // printf("%d\n", cell_id * 2 + 1 + 2*(depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1)  + min_y*(width-1)*(depth-1)));
            return cell_id * 2 + 1 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
        }

    }

    else if(direction == 3){
        int cell_id = min_z * (width - 1) * (height - 1) + min_y * (width - 1) + min_x;
        
        int current_id = -1;
        if(sortedV3_x == min_x && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1&& sortedV2_y == min_y && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 0;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y + 1 && sortedV1_z == min_z+1 
        ) {current_id = 1;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 2;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 3;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 4;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x + 1 && sortedV1_y== min_y + 1 && sortedV1_z == min_z +1 
        ) {current_id = 5;}
        id = cell_id * 6 + current_id + 2 * depth * (width - 1) * (height - 1) + 2 * width * (height - 1) * (depth - 1) +  2 * height * (width - 1) * (depth - 1);
        // printf("%d\n", id);
        return id;
        
    }
    return -1;
    

}

__device__ int getTriangleID_3d(int v1, int v2, int v3, int width, int height, int depth){
    // xy plane

    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    

    if(sortedV1_z == sortedV2_z && sortedV2_z == sortedV3_z)
    {
        int id = getTriangleID_3d_internel(sortedV1, sortedV2, sortedV3, width, height, depth) ;
        
        return id;
    }
    // yz plane
    if(sortedV1_x == sortedV2_x && sortedV2_x == sortedV3_x)
    {
        int id = getTriangleID_3d_internel(sortedV1, sortedV2, sortedV3, width, height, depth, 1) ;
        // printf("sortedV1: %d %d %d, sortedV2: %d %d %d, sortedV3: %d %d %d, id: %d \n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z, id + sortedV1_x * 2*(width-1)*(height-1) + depth * 2*(width-1)*(height-1) );
        return id;
    }

    
    
    if(sortedV1_y == sortedV2_y && sortedV2_y == sortedV3_y)
    {
        // if(sortedV1_x == 9 && sortedV1_y == 9 && sortedV1_z == 9) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
        int id = getTriangleID_3d_internel(sortedV1, sortedV2, sortedV3, width, height, depth, 2);
        // printf("sortedV1: %d %d %d, sortedV2: %d %d %d, sortedV3: %d %d %d, id: %d \n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z, id + sortedV1_y * 2*(width-1)*(height-1) + depth * 2*(width-1)*(height-1) + width * 2*(width-1)*(height-1) );
        return id;
    }
    
    else
    {
        int id = getTriangleID_3d_internel(sortedV1, sortedV2, sortedV3, width, height, depth, 3);
        // printf("sortedV1: %d %d %d, sortedV2: %d %d %d, sortedV3: %d %d %d, id: %d \n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z, id + sortedV1_y * 2*(width-1)*(height-1) + depth * 2*(width-1)*(height-1) + width * 2*(width-1)*(height-1) );
        return id;

    }

    return -1;
}





__global__ void get_vertex_traingle(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=num_Elements) return;

    int x = index % width;
    int y = (index / (width)) % height;
    int z = (index / (width * height)) % depth;
    int numTriangles = 0;
    for (int k = 0; k < maxNeighbors; k++) {
        int dx = directions[k * 3 + 0];
        int dy = directions[k * 3 + 1];
        int dz = directions[k * 3 + 2];

        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;

        if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) continue;
        int v2 = nx + ny * (width) + nz * (width * height);

        int v2_x = nx;
        int v2_y = ny;
        int v2_z = nz;
        for(int i=0;i<maxNeighbors;i++){
            int nx = x + directions[i * 3 + 0];
            int ny = y + directions[i * 3 + 1];
            int nz = z + directions[i * 3 + 2];
            for(int j=0;j<maxNeighbors;j++){
                int nx1 = v2_x + directions[j * 3 + 0];
                int ny1 = v2_y + directions[j * 3 + 1];
                int nz1 = v2_z + directions[j * 3 + 2];
                int neighbor = nx + ny * width + nz* (height * width);
                
                if(nx == nx1 && ny == ny1 && nz == nz1 && 
                    nx >=0 && nx < width && ny >= 0 & ny <height && nz >= 0 
                    && nz<depth && neighbor < num_Elements && neighbor >=0 
                    && neighbor!=index && neighbor != v2 && neighbor > v2){
                    if(index == 4830) printf("%d %d %d\n", index, v2, neighbor);
                    int triangleID = getTriangleID_3d(index, v2, neighbor, width, height, depth);
                    vertex_cells[index*42 + numTriangles++] = triangleID;
                }
            }
            // for (int j = 0; j < maxNeighbors; j++) {
            //     int dx = directions[j * 3 + 0];
            //     int dy = directions[j * 3 + 1];
            //     int dz = directions[j * 3 + 2];

            //     int nx = v2_x + dx;
            //     int ny = v2_y + dy;
            //     int nz = v2_z + dz;
            //     if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) continue;
            //     int v3 = nx + ny * (width) + nz * (width * height);
            //     int triangleID = getTriangleID(i, v2, v3, width, height, depth );
            //     vertex_cells[i*maxNeighbors + numTriangles++] = triangleID;
            // }
        }

        
    }
    
}

__global__ void ComputeDirection(int type = 0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;

    float *offset = input_data;
    int *DS_M_t = DS_M;
    int *AS_M_t = AS_M;
    if(type == 1){
        offset = decp_data;
        DS_M_t = dec_DS_M;
        AS_M_t = dec_AS_M;
    }

    int largest_neighbor = i;
    int smallest_neighbor = i;
    int x = i%width;
    int y = (i/width) % height;
    int z = (i / (width * height)) % depth;
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            // int neighbor_id = adjacency[maxNeighbors * i + j];
            // int r = neighbor_id;
            // if(neighbor_id == -1) continue;
            if(offset[largest_neighbor] < offset[r] || (offset[largest_neighbor] == offset[r] && largest_neighbor < r)) largest_neighbor = r;
            if(offset[smallest_neighbor] > offset[r] || (offset[smallest_neighbor] == offset[r] && smallest_neighbor > r)) smallest_neighbor = r;
        }
    }
    // if(i == 60111) std::cout<<"id is:"<<largest_neighbor<<std::endl;
    DS_M_t[i] = largest_neighbor;
    AS_M_t[i] = smallest_neighbor;
    
    // for(int j = 0; j< maxNeighbors; j++)
    // {
    //     int neighbor_id = adjacency[ maxNeighbors * i + j];
    //     if(neighbor_id == -1) continue;
        
    // }

    
    return;
};

__global__ void ComputeDescendingManifold(int type = 0){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
    
    int *DS_M_t = DS_M;
    
    if(type == 1){
        DS_M_t = dec_DS_M;
    }

    int v = i;
    bool is_saddle_neighbor = false;
    int x = i%width;
    int y = (i/width) % height;
    int z = (i / (width * height)) % depth;
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            // int neighbor_id = adjacency[maxNeighbors * i + j];
            // if(neighbor_id == -1) continue;
            if(vertex_type[r] == 2 || vertex_type[r] == 3){
                is_saddle_neighbor = true;
                break;
            }
        }
    }
    // for(int j = 0; j<maxNeighbors; j++){
        
    //     int neighborId = adjacency[i*maxNeighbors+j];
    //     if(neighborId == -1) continue;
    //     if(vertex_type[neighborId] == 2 || vertex_type[neighborId] == 3){
    //         is_saddle_neighbor = true;
    //         break;
    //     }
    // }

    if(!is_saddle_neighbor) return;
    
    while(true){
        int u = DS_M_t[v];
        int w = DS_M_t[u];
        if (u == w) break;
        DS_M_t[v] = w;
    }
    return;
}; 

__global__ void ComputeAscendingManifold(int type = 0){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;

    
    int *AS_M_t = AS_M;
    
    if(type == 1){
        AS_M_t = dec_AS_M;
    }

    int v = i;
    bool is_saddle_neighbor = false;
    // for(int j = 0; j<maxNeighbors; j++){
    //     int neighborId = adjacency[i*maxNeighbors+j];
    //     if(neighborId == -1) continue;
    //     if(vertex_type[neighborId] == 1 || vertex_type[neighborId] == 3){
    //         is_saddle_neighbor = true;
    //         break;
    //     }
    // }
    int x = i%width;
    int y = (i/width) % height;
    int z = (i / (width * height)) % depth;
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            // int neighbor_id = adjacency[maxNeighbors * i + j];
            // if(neighbor_id == -1) continue;
            if(vertex_type[r] == 1 || vertex_type[r] == 3){
                is_saddle_neighbor = true;
                break;
            }
        }
    }

    if(!is_saddle_neighbor) return;
    while(true){
        int u = AS_M_t[v];
        int w = AS_M_t[u];
        if (u == w) break;
        AS_M_t[v] = w;
    }
    return;
}; 

// __global__ void ComputeDirection(int direction = 0, int type = 0){
    
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i>=num_Elements) return;
//     float *offset = input_data;
//     int *M_t = DS_M;
//     if(direction == 1 && type == 0) M_t = AS_M;
//     else if(direction == 0 && type == 1) M_t = dec_DS_M;
//     else M_t = dec_AS_M;

//     if(type == 1) offset = decp_data;

//     if(direction == 0){
//         int largest_neighbor = i;
//         for(int j = 0; j< maxNeighbors; j++){
//             int neighbor_id = adjacency[ maxNeighbors * i + j];
//             if(neighbor_id == -1) continue;
//             if(offset[largest_neighbor] < offset[neighbor_id] || (offset[largest_neighbor] == offset[neighbor_id] && largest_neighbor < neighbor_id)) largest_neighbor = neighbor_id;
//         }
//         M_t[i] = largest_neighbor;
//     }

//     else{
//         int largest_neighbor = i;
//         for(int j = 0; j< maxNeighbors; j++)
//         {
//             int neighbor_id = adjacency[ maxNeighbors * i + j];
//             if(neighbor_id == -1) continue;
//             if(offset[largest_neighbor] > offset[neighbor_id] || (offset[largest_neighbor] == offset[neighbor_id] && largest_neighbor > neighbor_id)) largest_neighbor = neighbor_id;
//         }

//         M_t[i] = largest_neighbor;
//     }
    
    
    

    
//     return;
// }; 

__global__ void init_Manifold(int type=0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
    
    int *DS_M_t = DS_M;
    int *AS_M_t = AS_M;
    if(type == 1){
        DS_M_t = dec_DS_M;
        AS_M_t = dec_AS_M;
    }

    DS_M_t[i] = i;
    AS_M_t[i] = i;

}

__global__ void init_max_saddle(int type=0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=nMax) return;
    
    dec_reachable_saddle_for_Max[i*45 + 44] = 0;

}

__global__ void init_min_saddle(int type=0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=nMin) return;
    
    dec_reachable_saddle_for_Min[i*45 + 44] = 0;

}

// __global__ void init_min_saddle(int type=0){
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i>=nSaddle1) return;
    
//     dec_reachable_saddle_for_Min[i*(nSaddle1+1) + nSaddle1] = 0;

// }

__global__ void ExtractCP(int data_type = 0){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=num_Elements) return;

    int *minimum_t = minimum;
    int *saddles1_t = saddles1;
    int *saddles2_t = saddles2;
    int *maximum_t = maximum;
    int *nSaddle2_t = &nSaddle2;
    int *nSaddle1_t = &nSaddle1;
    int *nMax_t = &nMax;
    int *nMin_t = &nMin;
    if(data_type == 1){
        minimum_t = dec_minimum;
        saddles1_t = dec_saddles1;
        saddles2_t = dec_saddles2;
        maximum_t = dec_maximum;
        nSaddle2_t = &dec_nSaddle2;
        nSaddle1_t = &dec_nSaddle1;
        nMax_t = &dec_nMax;
        nMin_t = &dec_nMin;
    }
    int type = vertex_type[idx];
    int pos;
    if (type == 2 || type == 3) {
        pos = atomicAdd(nSaddle2_t, 1);
        saddles2_t[pos] = idx;
    }

    if (type == 1 || type == 3) {
        int pos = atomicAdd(nSaddle1_t, 1);
        saddles1_t[pos] = idx;
    }

    if (type == 4) {
        int pos = atomicAdd(nMax_t, 1);
        maximum_t[pos] = idx;
    }

    if (type == 0) {
        int pos = atomicAdd(nMin_t, 1);
        minimum_t[pos] = idx;
    }

    // if(type==-1) printf("did not classified : %d\n", idx);
}

__global__ void ComputeTempArray(int direction = 0, int type = 0){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if((idx>=nMin && direction == 0) || (idx>=nMax && direction == 1) || (idx>=nSaddle1 && direction == 2) || (idx>=nSaddle2 && direction == 3)) return;

    int *tempArray_t = dec_tempArray;
    int *tempArrayMin_t = dec_tempArrayMin;
    int *saddleindex_t = dec_saddlerank;
    int *saddle1index_t = dec_saddle1rank;
    int *minimum_t = dec_minimum;
    int *maximum_t = dec_maximum;
    int *saddles2_t = dec_saddles2;
    int *saddles1_t = dec_saddles1;
    if(type!=0){
        tempArray_t = tempArray;
        tempArrayMin_t = tempArrayMin;
        minimum_t = minimum;
        maximum_t = maximum;
        saddleindex_t = saddlerank;
        saddle1index_t = saddle1rank;
        saddles2_t = saddles2;
        saddles1_t = saddles1;

    }
    if(direction == 0) tempArrayMin_t[minimum_t[idx]] = idx;
    else if(direction == 1){
        tempArray_t[maximum_t[idx]] = idx;
    }
    else if(direction == 2){
        saddle1index_t[saddles1_t[idx]] = idx;
    }
    else{
        saddleindex_t[saddles2_t[idx]] = idx;
    }
    
}

__device__ bool islarger(const int v, const int u, const float *offset){
    return offset[v] > offset[u] || (abs(offset[v] - offset[u]) == 0 && v > u);
}

__device__ bool islarger_shared(const int v, const int u, float value_v1, float value_v2){
    return value_v1 > value_v2 || (value_v1 == value_v2 && v > u);
}

__device__ bool isless(const int v, const int u, const float* offset){
    return offset[v] < offset[u] || (std::abs(offset[v] - offset[u]) == 0 && v < u);
}

__device__ bool isless_shared(const int v, const int u, float value_v1, float value_v2){
    return value_v1 < value_v2 || (value_v1 == value_v2 && v < u);
}

__device__ void insertionSort(int *arr, int n, float *offset){
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;

        /* Move elements of arr[0..i-1], that are
           greater than key, to one position ahead
           of their current position */
        
        while (j >= 0 && isless(arr[j], key, offset)) {
            
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

__device__ void reversed_insertionSort(int *arr, int n, float *offset){
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;

        /* Move elements of arr[0..i-1], that are
           greater than key, to one position ahead
           of their current position */
        
        while (j >= 0 && islarger(arr[j], key, offset)) {
            
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}


__global__ void findAscPaths(int type = 0){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle2) return;

    int *saddles2_t = saddles2;
    int *saddleTriplets_t = dec_saddleTriplets;
    int *desManifold = dec_DS_M;
    float *offset = decp_data;
    int *tempArray_t = dec_tempArray;
    int *reachable_saddle_for_Max_t = dec_reachable_saddle_for_Max;
    int temp_Saddle_tripplets[46];

    if(type == 1){
        saddles2_t = saddles2;
        saddleTriplets_t = saddleTriplets;
        desManifold = DS_M;
        offset = input_data;
        tempArray_t = tempArray;
        reachable_saddle_for_Max_t = reachable_saddle_for_Max;
    }

    if(type == 1){
        decp_data[saddles2[index]] = input_data[saddles2[index]] - bound;
        delta_counter[saddles2[index]] = 7;
    }
    
    
    temp_Saddle_tripplets[44] = 0;
    const int vertexId = saddles2[index];
    int x = vertexId%width;
    int y = (vertexId/width) % height;
    int z = (vertexId / (width * height)) % depth;
    float vertexValue = offset[vertexId];
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            // int neighbor_id = adjacency[maxNeighbors * i + j];
            // if(neighbor_id == -1) continue;
            int maxi = desManifold[r];
            if(islarger(r, vertexId, offset) && abs(offset[maxi] - vertexValue) >= thr) {
                temp_Saddle_tripplets[temp_Saddle_tripplets[44]] = desManifold[r];
                temp_Saddle_tripplets[44]++;
            }
        }
    }
   
    temp_Saddle_tripplets[45] = saddles2_t[index];

   
    int ItemSize = temp_Saddle_tripplets[44];

    
    insertionSort(temp_Saddle_tripplets, ItemSize, offset);
    for(int i = 0; i < ItemSize; i++){
        saddleTriplets_t[index * 46 + i] = temp_Saddle_tripplets[i];
        // int max_index_t = max_index[temp_Saddle_tripplets[i]];
        // int idx_fp = atomicAdd(&reachable_saddle_for_Max_t[max_index_t * 45 + 44], 1);
        // reachable_saddle_for_Max_t[max_index_t * 45 + idx_fp] = saddles2[index];
        // wrong_neighbors_index[idx_fp] = neighborId;
    }
    saddleTriplets_t[index * 46 + 44] = temp_Saddle_tripplets[44];
    saddleTriplets_t[index * 46 + 45] = temp_Saddle_tripplets[45];
    // if(index == 0){
    //     for(int i = 0; i < ItemSize; i++){
    //     printf("%d %d\n", saddleTriplets[index * 46 + i] ,temp_Saddle_tripplets[i]);
    // }
    // }
    
}

__global__ void computeMaxIndex(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nMax) return;
    max_index[maximum[index]] = index;
    
}

__global__ void computeMinIndex(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nMin) return;
    max_index[minimum[index]] = index;
    
}



__global__ void findDescPaths(int type = 0){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle1) return;
    int *saddles1_t = saddles1;
    int *saddle1Triplets_t = dec_saddle1Triplets;
    int *asManifold = dec_AS_M;
    float *offset = decp_data;
    int *tempArray_t = dec_tempArrayMin;
    int temp_Saddle_tripplets[46];
    int *reachable_saddle_for_Min_t = dec_reachable_saddle_for_Min;

    if(type == 1){
        
        saddle1Triplets_t = saddle1Triplets;
        asManifold = AS_M;
        offset = input_data;
        tempArray_t = tempArrayMin;
        reachable_saddle_for_Min_t = reachable_saddle_for_Min;
    }

    if(type == 1){
        decp_data[saddles1[index]] = input_data[saddles1[index]] - bound;
        delta_counter[saddles1[index]] = 7;
    }
    
    temp_Saddle_tripplets[44] = 0;
    const int vertexId = saddles1_t[index];
    int x = vertexId%width;
    int y = (vertexId/width) % height;
    int z = (vertexId / (width * height)) % depth;
    float vertexValue = offset[vertexId];
    for(int j = 0; j< maxNeighbors; j++){

        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            int mini = asManifold[r];
            if(isless(r, vertexId, offset) && abs(offset[mini] - vertexValue) >= thr) {
                temp_Saddle_tripplets[temp_Saddle_tripplets[44]] = asManifold[r];
                temp_Saddle_tripplets[44]++;
            }
        }
    }
    // for(int k = 0; k < maxNeighbors; k++){
    //     const int neighborId = adjacency[maxNeighbors * vertexId + k];
        
    //     if(neighborId == -1) continue;
    //     if(isless(neighborId, vertexId, offset)) {
    //         temp_Saddle_tripplets[temp_Saddle_tripplets[44]] = asManifold[neighborId];
    //         temp_Saddle_tripplets[44]++;
    //     }
        
    // }

    int ItemSize = temp_Saddle_tripplets[44];
    temp_Saddle_tripplets[45] = saddles1_t[index];
    reversed_insertionSort(temp_Saddle_tripplets, ItemSize, offset);

    // for(int i = 0; i < 46; i++){
    //     saddle1Triplets_t[index * 46 + i] = temp_Saddle_tripplets[i];
    // }

    for(int i = 0; i < ItemSize; i++){
        saddle1Triplets_t[index * 46 + i] = temp_Saddle_tripplets[i];
        // int min_index_t = max_index[temp_Saddle_tripplets[i]];
        // int idx_fp = atomicAdd(&reachable_saddle_for_Min_t[min_index_t * 45 + 44], 1);
        // reachable_saddle_for_Min_t[min_index_t * 45 + idx_fp] = saddles1[index];
        // wrong_neighbors_index[idx_fp] = neighborId;
    }
    saddle1Triplets_t[index * 46 + 44] = temp_Saddle_tripplets[44];
    saddle1Triplets_t[index * 46 + 45] = temp_Saddle_tripplets[45];
    
}

__global__ void computelargestSaddlesForMax(int type = 0){
    // saddle is stored by descending rank;
    // find the largest saddle connected with eahc max;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nMax) return;
    int *saddleTriplets_t = saddleTriplets;
    int *saddlerank_t = saddlerank;
    int *AS_M_t = AS_M;
    int *largestSaddlesForMax_t = largestSaddlesForMax;
    int *reachable_saddle_for_Max_t = reachable_saddle_for_Max;
    float *offset = input_data;
    if(type!=0){
        saddleTriplets_t = dec_saddleTriplets;
        saddlerank_t = dec_saddlerank;
        AS_M_t = dec_AS_M;
        largestSaddlesForMax_t = dec_largestSaddlesForMax;
        offset = decp_data;
        reachable_saddle_for_Max_t = dec_reachable_saddle_for_Max;
    }

    largestSaddlesForMax_t[index] = -1;
    
    int globalMax = nMax - 1;
    int maxId = maximum[index];
    for(int i = 0; i < nSaddle2; i++) {
        // auto &triplet = saddleTriplets[i];
        const int saddle = saddles2[i];
        int temp;
        for(int p = 0; p < saddleTriplets_t[i * 46 + 44]; p++) {
            
            const auto &max = saddleTriplets_t[i * 46 + p];
            if(max != maxId) continue;
            
            if(largestSaddlesForMax_t[index] == -1){
                largestSaddlesForMax_t[index] = saddle;
                continue;
            }
            if(max != globalMax) {
                temp = largestSaddlesForMax_t[index];
                if(isless(temp, saddle,  offset)) {
                    largestSaddlesForMax_t[index]
                        = saddle;
                }
            }
        }
    }

    // int largest = -1;
    // int max_index_t = max_index[maxId];
    // int number_of_saddle = reachable_saddle_for_Max_t[max_index_t * 45 + 44] ;
    // for(int i = 0; i<number_of_saddle; i++){
        
    //     int reachable_saddle = reachable_saddle_for_Max_t[max_index_t * 45 + i];
    //     if(vertex_type[reachable_saddle] == 2 || vertex_type[reachable_saddle] == 3){
    //         if(largest == -1) largest = reachable_saddle;
    //         else if(isless(largest, reachable_saddle,  offset)) largest = reachable_saddle;
    //     }
    // }

    // if(largest != -1){
    //     // printf("wrong here %d %d %d\n", index, largest, largestSaddlesForMax_t[index] );
    //     largestSaddlesForMax_t[index] = largest;
    // }
}

__global__ void computesmallestSaddlesForMin(int type = 0){
    // saddle is stored by descending rank;
    // find the largest saddle connected with eahc max;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nMin) return;

    int *saddle1Triplets_t = saddle1Triplets;
    int *smallestSaddlesForMin_t = smallestSaddlesForMin;
    int *saddle1rank_t = saddle1rank;
    float *offset = input_data;
    int *reachable_saddle_for_Min_t = reachable_saddle_for_Min;
    if(type!=0){
        saddle1Triplets_t = dec_saddle1Triplets;
        smallestSaddlesForMin_t = dec_smallestSaddlesForMin;
        saddle1rank_t = dec_saddle1rank;
        offset = decp_data;
        reachable_saddle_for_Min_t = dec_reachable_saddle_for_Min;
    }

    smallestSaddlesForMin_t[index] = -1;
    int globalMin = nMin - 1;
    int minId = minimum[index];
    // for(int i = 0; i < nSaddle1; i++) {
    //     // auto &triplet = saddleTriplets[i];
    //     int saddle = saddles1[i];
    //     int temp;
    //     for(int p = 0; p < saddle1Triplets_t[i * 46 + 44]; p++) {
            
    //         const auto &min_v = saddle1Triplets_t[i * 46 + p];
    //         if(min_v != minId) continue;

    //         if(smallestSaddlesForMin_t[index] == -1){
    //             smallestSaddlesForMin_t[index] = saddle;
    //             continue;
    //         }
    //         if(min_v != globalMin) {
    //             temp = smallestSaddlesForMin_t[index];
    //             if(isless(saddle, temp, offset)) {
    //                 smallestSaddlesForMin_t[index] = saddle;
    //             }
    //         }
    //     }
    // }

    int smallest = -1;
    int min_index_t = max_index[minId];
    int number_of_saddle = reachable_saddle_for_Min_t[min_index_t * 45 + 44] ;
    for(int i = 0; i<number_of_saddle; i++){
        
        int reachable_saddle = reachable_saddle_for_Min_t[min_index_t * 45 + i];
        if(vertex_type[reachable_saddle] == 1 || vertex_type[reachable_saddle] == 3){
            if(smallest == -1) smallest = reachable_saddle;
            else if(islarger(smallest, reachable_saddle,  offset)) smallest = reachable_saddle;
        }
    }

    if(smallest != -1){
        // printf("wrong here %d %d %d\n", index, smallest, smallestSaddlesForMin_t[index] );
        smallestSaddlesForMin_t[index] = smallest;
    }
}

__device__ int computeMaxLabel(int i, const float *offset){
        
    int current_id = i;
    int largest_neighbor = current_id;  
    int next_largest_neighbor;
    while (true) {
        
        next_largest_neighbor = largest_neighbor;
        
        // if(i == 880) printf("%d\n", next_largest_neighbor);
        for (int j = 0; j < maxNeighbors; j++) {
            int neighbor_id = adjacency[maxNeighbors * current_id + j];
            if (neighbor_id == -1) continue;  

            
            if (offset[next_largest_neighbor] < offset[neighbor_id] || 
            (offset[next_largest_neighbor] == offset[neighbor_id] && next_largest_neighbor < neighbor_id)) {
                next_largest_neighbor = neighbor_id;
            }
        }

        if (next_largest_neighbor == largest_neighbor) break;

        current_id = next_largest_neighbor;
        largest_neighbor = next_largest_neighbor;
    }

    return current_id;

}

__device__ int computeMinLabel(int i, const float *offset){
        
    int current_id = i;
    int largest_neighbor = current_id;  

    while (true) {
        int next_largest_neighbor = largest_neighbor;
        
        for (int j = 0; j < maxNeighbors; j++) {
            int neighbor_id = adjacency[maxNeighbors * current_id + j];
            if (neighbor_id == -1) continue;  

            
            if (offset[next_largest_neighbor] > offset[neighbor_id] || 
            (offset[next_largest_neighbor] == offset[neighbor_id] && next_largest_neighbor > neighbor_id)) {
                next_largest_neighbor = neighbor_id;
            }
        }

        if (next_largest_neighbor == largest_neighbor) break;

        current_id = next_largest_neighbor;
        largest_neighbor = next_largest_neighbor;
    }

    return current_id;

}

__global__ void compute_Max_for_Saddle(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle2) return;
    int saddle = saddles2[index];
    int label_count = 0;
    for(int j = 0; j< maxNeighbors; j++){
        int neighborId = adjacency[maxNeighbors * saddle + j];
        if(neighborId == -1) continue;
        if(islarger(neighborId, saddle, input_data)){
            
            int l = computeMaxLabel(neighborId, input_data);
            or_saddle_max_map[index * 4 + label_count] = l;
            label_count++;
        }

    }
    return;
}

__global__ void compute_Min_for_Saddle(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle1) return;
    int label_count = 0;
    int saddle = saddles1[index];
    for(int j = 0; j< maxNeighbors; j++){
        int neighborId = adjacency[ maxNeighbors * saddle + j];
        if(neighborId == -1) continue;
        if(isless(neighborId, saddle, input_data)){
            int l = computeMinLabel(neighborId, input_data);
            or_saddle_min_map[index * 4 + label_count] = l;
            label_count++;
        }

    }
    return;
}

__global__ void get_wrong_split_neighbors(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle2) return;
    int label_count = 0;

    int saddle = saddles2[index];
    int x = saddle%width;
    int y = (saddle/width) % height;
    int z = (saddle/ (width * height)) % depth;
    float vertexValue = decp_data[saddle];
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int neighborId = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && neighborId >= 0 && neighborId < width*height*depth && newZ<depth && newZ>=0) {
            // int neighbor_id = adjacency[maxNeighbors * i + j];
            // if(neighbor_id == -1) continue;
            int l = dec_DS_M[neighborId];
            // if(islarger(neighborId, saddle, decp_data) && abs(decp_data[l] - vertexValue) >= thr){
            if(islarger(neighborId, saddle, decp_data)){
                if(l!= DS_M[neighborId]){
                    if(wrong_neighbors[neighborId] == 0){
                        int idx_fp = atomicAdd(&num_false_cases, 1);
                        wrong_neighbors[neighborId] = 1;
                        wrong_neighbors_index[idx_fp] = neighborId;
                        return;
                    }
                }
                label_count++;
            }
        }
    }
    // for(int j = 0; j< maxNeighbors; j++){
    //     int neighborId = adjacency[ maxNeighbors * saddle + j];
    //     if(neighborId == -1) continue;
    //     if(islarger(neighborId, saddle, decp_data)){
            
    //         int l = dec_DS_M[neighborId];
            
    //         if(l!= DS_M[neighborId]){
    //             if(wrong_neighbors[neighborId] == 0){
    //                 int idx_fp = atomicAdd(&num_false_cases, 1);
    //                 wrong_neighbors[neighborId] = 1;
    //                 wrong_neighbors_index[idx_fp] = neighborId;
    //                 return;
    //             }
    //         }
    //         label_count++;
    //     }

    // }
}

__global__ void get_wrong_join_neighbors(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=nSaddle1) return;
    int label_count = 0;

    int saddle = saddles1[index];
    int x = saddle%width;
    int y = (saddle/width) % height;
    int z = (saddle / (width * height)) % depth;
    float vertexValue = decp_data[saddle];
    for(int j = 0; j< maxNeighbors; j++){
        int dirX = directions[j * 3];     
        int dirY = directions[j * 3 + 1]; 
        int dirZ = directions[j * 3 + 2]; 
        int newX = x + dirX;
        int newY = y + dirY;
        int newZ = z + dirZ;
        int neighborId = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && neighborId >= 0 && neighborId < width*height*depth && newZ<depth && newZ>=0) {
            int l = dec_AS_M[neighborId];
            // if(isless(neighborId, saddle, decp_data) && abs(decp_data[l] - vertexValue) >= thr){
            if(isless(neighborId, saddle, decp_data)){
                
                if(l!=AS_M[neighborId]){
                    if(wrong_neighbors_ds[neighborId] == 0){
                        int idx_fp = atomicAdd(&num_false_cases1, 1);
                        wrong_neighbors_ds[neighborId] = 1;
                        wrong_neighbors_ds_index[idx_fp] = neighborId;
                        return;
                    }
                }
                label_count++;
        }
    }
    }
    // for(int j = 0; j< maxNeighbors; j++){
    //     int neighborId = adjacency[ maxNeighbors * saddle + j];
    //     if(neighborId == -1) continue;

    //     if(isless(neighborId, saddle, decp_data)){
            
    //         int l = dec_AS_M[neighborId];
    //         if(l!=AS_M[neighborId]){
    //             if(wrong_neighbors_ds[neighborId] == 0){
    //                 int idx_fp = atomicAdd(&num_false_cases1, 1);
    //                 wrong_neighbors_ds[neighborId] = 1;
    //                 wrong_neighbors_ds_index[idx_fp] = neighborId;
    //                 return;
    //             }
    //         }
    //         label_count++;
    //     }

    // }
}



__global__  void get_false_criticle_points(){

    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=num_Elements) return;
   
    int type1 = dec_vertex_type[i];
    
    if(type1 != vertex_type[i]){
        
        // maximum
        if((type1==4 and vertex_type[i]!=4) or (type1!=4 and vertex_type[i]==4)){
            int idx_fp_max = atomicAdd(&count_f_max, 1);
            all_max[idx_fp_max] = i;
            
        } 
        if((type1==0 and vertex_type[i]!=0) or (type1!=0 and vertex_type[i]==0)){
            int idx_fp_min = atomicAdd(&count_f_min, 1);
            all_min[idx_fp_min] = i;
        }
        if((type1==2 and vertex_type[i]!=2) or (type1!=2 and vertex_type[i]==2)){
            int idx_fp_saddle = atomicAdd(&count_f_saddle, 1);
            all_saddle[idx_fp_saddle] = i;
        }

        if((type1==1 and vertex_type[i]!=1) or (type1!=1 and vertex_type[i]==1)){
            int idx_fp_saddle = atomicAdd(&count_f_saddle, 1);
            all_saddle[idx_fp_saddle] = i;
        }

        if((type1==3 and vertex_type[i]!=3) or (type1!=3 and vertex_type[i]==3)){
            int idx_fp_saddle = atomicAdd(&count_f_saddle, 1);
            all_saddle[idx_fp_saddle] = i;
        }


    }

    else if(type1==2 and vertex_type[i]==2 || type1== 1 and vertex_type[i]==1 || type1== 3 and vertex_type[i]==3){
            
        if(areVectorsDifferent(i, 0)){
            int idx_fp_saddle = atomicAdd(&count_f_saddle, 1);
            all_saddle[idx_fp_saddle] = i;
        }
    }
        
        
    

}


__global__ void init_delta() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num_Elements){
        d_deltaBuffer[tid] = -4.0 * bound;
        // delta_counter[tid] = 1;
        updated_vertex[tid] = -1;
    }

}

__global__ void init_update() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num_Elements){
        updated_vertex[tid] = -1;
    }

}

__global__ void init_neighbor_buffer() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num_Elements){
        wrong_neighbors[tid] = 0;
        wrong_neighbors_ds[tid] = 0;
    }

}

__global__ void init_buffer(int type = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num_Elements){
        if(type == 0){
            wrong_rank_max[tid] = 0;
            wrong_rank_max_2[tid] = 0;
        }
        else{
            wrong_rank_min[tid] = 0;
            wrong_rank_min_2[tid] = 0;
        }
        
    }

}

__global__ void init_saddle_buffer(int type = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num_Elements){
        if(type == 0){
            wrong_neighbors[tid] = 0;
            wrong_neighbors_ds[tid] = 0;
        }
        else{
            wrong_rank_min[tid] = 0;
            wrong_rank_min_2[tid] = 0;
        }
        
    }

}

__global__ void init_saddle_rank_buffer(int type = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num_Elements){
        if(type == 0){
            wrong_rank_saddle[tid] = 0;
        }
        else{
            wrong_rank_saddle_join[tid] = 0;
        }
        
    }

}


__device__ float atomicCASfloat(float* address, float val) {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old_val_as_ui = *address_as_ui;
    unsigned int new_val_as_ui = __float_as_int(val);
    unsigned int assumed;

    assumed = old_val_as_ui;
    old_val_as_ui = atomicCAS(address_as_ui, assumed, new_val_as_ui);

    return __int_as_float(old_val_as_ui);
}

__device__ int swap1(int index, float diff){
    int update_successful = 0;
    float old_value = d_deltaBuffer[index];
    while (update_successful==0) {
        float current_value = d_deltaBuffer[index];
        if (diff > current_value) {
            float swapped = atomicCASfloat(&d_deltaBuffer[index], diff);
            if (swapped == current_value) {
                update_successful = 1;
                
            } else {
                old_value = swapped;
            }
        } else {
            update_successful = 1; 
        }
    }
    
}


__device__ int swap(int index, float diff){
    
    float old_value = d_deltaBuffer[index];
    if ( diff > old_value) {                    
        swap1(index, diff);
    } 
    
    
    // else d_deltaBuffer[index] = diff;
}

__device__ int findDifferences(const int* vec1, int size1, 
                               const int* vec2, int size2, 
                               int index, int* differences) {
    int count = 0;
    
    
    for (int i = 0; i < size2; i++) {
        int val = vec2[index * (maxNeighbors + 1) + i];
        bool found = false;

        // 遍历 `vec1`，检查 `val` 是否存在
        for (int j = 0; j < size1; j++) {
            if (vec1[index * (maxNeighbors + 1) + j] == val) {
                found = true;
                break;
            }
        }

        // 如果 `val` 不在 `vec1`，则存入 `differences`
        if (!found) {
            differences[count++] = val;
            
        }
    }
    return count; // 返回 `differences` 数组的大小
}

__device__ int findDifferences_local(const int* vec1, int size1, const int* vec2, int size2, int* differences) {
    int count = 0;
    
    for (int i = 0; i < size2; i++) {
        int val = vec2[i];
        bool found = false;

        // 遍历 `vec1`，检查 `val` 是否存在
        for (int j = 0; j < size1; j++) {
            if (vec1[j] == val) {
                found = true;
                break;
            }
        }

        // 如果 `val` 不在 `vec1`，则存入 `differences`
        if (!found) {
            differences[count++] = val;
            
        }
    }
    return count; // 返回 `differences` 数组的大小
}


__device__ void fix_saddle(int i){
    
    
    float delta;
    
    if(areVectorsDifferent(i, 0)){
        
        int diff[MAX_NEIGHBORS];
        
        int *lowerStar_start = dec_lowerStars;
        int size1 = dec_lowerStars[(maxNeighbors + 1) * i + maxNeighbors];

        int *o_lowerStar_start = lowerStars ;
        int o_size1 = lowerStars[(maxNeighbors + 1) * i + maxNeighbors];
        int count = findDifferences(lowerStar_start, size1, o_lowerStar_start, o_size1, i, diff);
        
        for(int index = 0; index<count; index++){
            const int id = diff[index];
            int c = delta_counter[id] + 1;
            delta = -bound / (1 << 2);
            // delta = (input_data[id]-bound) - decp_data[id];
            float oldValue = d_deltaBuffer[id];
            
            if (delta > oldValue) {
                swap(id, delta);
            }  

        }
        
        int diff1[MAX_NEIGHBORS];
        
        count = findDifferences(o_lowerStar_start, o_size1, lowerStar_start, size1, i, diff1);
        if(count > 0){
            // delta = (input_data[i]-bound) - decp_data[i];
            int c = delta_counter[i] + 1;
            delta = -bound / (1 << 2);
            float oldValue = d_deltaBuffer[i];
            if (delta > oldValue) {
                swap(i, delta);
            } 
        }
    
    }
    return;
}

__device__ void fix_saddle_local(int i, int *o_lowerStar_start, int o_size1){
    
    
    float delta;
    
    // if(areVectorsDifferent(i, 0)){

    int lowerCount = 0, upperCount = 0;
    int lowerStar[MAX_NEIGHBORS] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
   

    int gx = i % width;
    int gy = (i/width) % height;
    int gz = (i/(width*height)) % depth;
    int neighbor_size = 0;
    float currentHeight = decp_data[i];
    for (int d = 0; d < maxNeighbors; d++) {
        
        int dirX = directions[d * 3];     
        int dirY = directions[d * 3 + 1]; 
        int dirZ = directions[d * 3 + 2]; 
        int newX = gx + dirX;
        int newY = gy + dirY;
        int newZ = gz + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
            
            
            float neighbor_value = decp_data[r];

            if (neighbor_value < currentHeight || (neighbor_value == currentHeight && r < i)) {
                
                lowerStar[lowerCount] = r;
                lowerCount++;
                
            } 
        }
    }
    int diff[MAX_NEIGHBORS];
    
    
    int count = findDifferences_local(lowerStar, lowerCount, o_lowerStar_start, o_size1, diff);
    
    for(int index = 0; index<count; index++){
        const int id = diff[index];
        int c = delta_counter[index] + 1;
        delta = -bound / (1 << 2);
        // delta = (input_data[id]-bound) - decp_data[id];
        float oldValue = d_deltaBuffer[id];
        
        if (delta > oldValue) {
            swap(id, delta);
        }  

    }
    
    int diff1[MAX_NEIGHBORS];
    
    count = findDifferences_local(o_lowerStar_start, o_size1, lowerStar, lowerCount, diff1);
    if(count > 0){
        int c = delta_counter[i] + 1;
        delta = -bound / (1 << 2);
        // delta = (input_data[i]-bound) - decp_data[i];
        float oldValue = d_deltaBuffer[i];
        if (delta > oldValue) {
            swap(i, delta);
        } 
    }
    
    // }
    return;
}

__global__ void c_loop(int direction = 0){      
    // preservation of split tree?->decrease f
    
    if (direction == 0){
        
        // if vertex is a regular point.
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i>=count_f_max) return;
        
        int index = all_max[i];
        int x = index % width;
        int y = (index / width) % height;
        int z = (index / (width * height)) % depth;
        float input_value = input_data[index];
        float decp_value = decp_data[index];
        if (vertex_type[index]!=4){
            int c = delta_counter[index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((input_value - bound) + decp_value) / 2.0 - decp_value;

            float oldValue = d_deltaBuffer[index];
            
            if (d > oldValue) {
                swap(index, d);
            }  

            return;
        
        }
        else{
            
            // if is a maximum in the original data;
            int largest_index = index;
            float largest_value = decp_data[largest_index];
            for(int i = 0; i< maxNeighbors;i++){
                int dx = directions[3 * i];
                int dy = directions[3 * i + 1];
                int dz = directions[3 * i + 2];

                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                int neighbor = x + dx + (y + dy + (z + dz) * height) * width;
                if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth || neighbor < 0 || neighbor >= num_Elements) continue;
                
                // neighbor = adjacency[maxNeighbors * index + i];
                // if(neighbor == -1) continue;
                float neighbor_value = decp_data[neighbor];
                if(islarger_shared(neighbor, largest_index, neighbor_value, largest_value))
                {
                    largest_index = neighbor;
                    largest_value = decp_data[largest_index];
                }
            }
            
            if(decp_value>largest_value or(decp_value==largest_value and index>largest_index)){
                return;
            }
            int c = delta_counter[largest_index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((input_data[largest_index] - bound) + largest_value) / 2.0 - largest_value;
            
            float oldValue = d_deltaBuffer[largest_index];
            if (d > oldValue) {
                swap(largest_index, d);
            }  

            return;
        }
    }
    
    else if (direction == 1){
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i>=count_f_min) return;
        
        int index = all_min[i];
        int x = index % width;
        int y = (index / width) % height;
        int z = (index / (width * height)) % depth;
        float input_value = input_data[index];
        float decp_value = decp_data[index];
        if (vertex_type[index]!=0){
            int smallest_index = index;
            float smallest_value = input_data[smallest_index];
            for(int i = 0; i< maxNeighbors;i++){
                int dx = directions[3 * i];
                int dy = directions[3 * i + 1];
                int dz = directions[3 * i + 2];

                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                int neighbor = x + dx + (y + dy + (z + dz) * height) * width;
                if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth || neighbor<0 || neighbor >= num_Elements) continue;
                
                // neighbor = adjacency[maxNeighbors * index + i];
                // if(neighbor == -1) continue;
                float neighbor_value = input_data[neighbor];
                if(isless_shared(neighbor, smallest_index, neighbor_value, smallest_value)){
                    smallest_index = neighbor;
                    smallest_value = input_data[smallest_index];
                }
            }
            // for(int i = 0; i<maxNeighbors;i++){
            //     int neighbor = adjacency[maxNeighbors * index + i];
            //     if(neighbor == -1) continue;
            //     if(isless(neighbor, smallest_index, input_data))
            //     {
            //         smallest_index = neighbor;
            //     }
            // }
            float decp_smallest_value = decp_data[smallest_index];
            int c = delta_counter[smallest_index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((smallest_value - bound) + decp_smallest_value) / 2.0 - decp_smallest_value;
            if(decp_value>decp_smallest_value or (decp_value==decp_smallest_value and index>smallest_index)){
                return;
            }
            float oldValue = d_deltaBuffer[smallest_index];
            if (d > oldValue) {
                swap(smallest_index, d);
            }  
            return;
        
        }
    
        else{
            int c = delta_counter[index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((input_value - bound) + decp_value) / 2.0 - decp_value;
            float oldValue = d_deltaBuffer[index];
            if (d > oldValue) {
                swap(index, d);
            } 
            return;
        }
    }    

    else{

        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i>=count_f_saddle) return;
        
        int index = all_saddle[i];
        // fix_saddle(index);
        fix_saddle_local(index, lowerStars+index*(maxNeighbors + 1), lowerStars[index*(maxNeighbors + 1) +maxNeighbors]);
        // fix_saddle_local(int i, int *lowerStar_start, int size1, int *o_lowerStar_start, int o_size1)
    }

    
    return;
}





__global__ void applyDeltaBuffer() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_Elements) {
        
        if(d_deltaBuffer[tid]!=-4.0 * bound){
            if(abs(d_deltaBuffer[tid])>1e-15 && delta_counter[tid]<6 && abs(input_data[tid]-(decp_data[tid] + d_deltaBuffer[tid])) <= bound){
                decp_data[tid] += d_deltaBuffer[tid]; 
                delta_counter[tid]+=1;
            }
            else{
                delta_counter[tid] = 7;
                decp_data[tid] = input_data[tid] - bound;
            }
            int x = tid % width;
            int y = (tid  / width) % height;
            int z = (tid  / (width * height)) % depth;
            for(int i = 0; i< maxNeighbors;i++){
                int dx = directions[3 * i];
                int dy = directions[3 * i + 1];
                int dz = directions[3 * i + 2];

                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                
                int neighborId = x + dx + (y + dy + (z + dz) * height) * width;
                // if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) continue;
                if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth || neighborId < 0 || neighborId >= num_Elements) continue;
                // neighborId = adjacency[maxNeighbors * tid + i];
                // if(neighborId == -1) continue;
                updated_vertex[neighborId] = 0;
            }
            // for(int i = 0;i<14; i++){
            //     int neighborId = adjacency[14*tid + i];
            //     if(neighborId == -1) break;
            //     updated_vertex[neighborId] = 0;
            // }
            updated_vertex[tid] = 0;
        }     
    }
        
}

__global__ void get_wrong_index_max(){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=nSaddle2 ||saddleTriplets[i*46 + 44] == 0) return;
    
    if(saddleTriplets[i*46] != dec_saddleTriplets[i*46]){
        
        int maxId = saddleTriplets[i*46];
        if(wrong_rank_max[maxId] == 0){
            
            int pos = atomicAdd(&wrong_max_counter, 1);
            wrong_rank_max_index[pos * 2] = saddleTriplets[i*46];
            wrong_rank_max_index[pos * 2 + 1] = dec_saddleTriplets[i*46];
            wrong_rank_max[maxId] = 1;
        }
    }
    int numberOfMax = saddleTriplets[i * 46 + 44] - 1;

    // if(saddleTriplets[i * 46 + numberOfMax] != dec_saddleTriplets[i * 46 + numberOfMax]){
    //     int maxId = saddleTriplets[i * 46 + numberOfMax];
    //     if(wrong_rank_max_2[maxId] == 0){
    //         int pos = atomicAdd(&wrong_max_counter_2, 1);
    //         wrong_rank_max_index_2[pos * 2] = saddleTriplets[i * 46 + numberOfMax];
    //         wrong_rank_max_index_2[pos * 2 + 1] = dec_saddleTriplets[i * 46 + numberOfMax];
    //         wrong_rank_max_2[maxId] = 1;
            
            
    //     }
    // }
        
    
}

__global__ void get_wrong_index_min(){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=nSaddle1 || saddle1Triplets[i*46 + 44] == 0) return;
    if(saddle1Triplets[i*46] != dec_saddle1Triplets[i*46]){
        
        int minId = saddle1Triplets[i*46];
        if(wrong_rank_min[minId] == 0){
            int pos = atomicAdd(&wrong_min_counter, 1);
            wrong_rank_min_index[pos * 2] = saddle1Triplets[i*46];
            wrong_rank_min_index[pos * 2 + 1] = dec_saddle1Triplets[i*46];
            wrong_rank_min[minId] = 1;
            
        }
    }
    int numberOfmin = saddle1Triplets[i * 46 + 44] - 1;
    int dec_numberOfmin = dec_saddle1Triplets[i * 46 + 44] - 1;
    // if(saddle1Triplets[i * 46 + numberOfmin] != dec_saddle1Triplets[i * 46 + dec_numberOfmin]){
    //     int minId = saddle1Triplets[i * 46 + numberOfmin];
    //     if(wrong_rank_min_2[minId] == 0){
            
    //         int pos = atomicAdd(&wrong_min_counter_2, 1);
    //         wrong_rank_min_index_2[pos * 2] = saddle1Triplets[i * 46 + numberOfmin];
    //         wrong_rank_min_index_2[pos * 2 + 1] = dec_saddle1Triplets[i * 46 + dec_numberOfmin];
    //         wrong_rank_min_2[minId] = 1;
    //     }
    // }
        
    
}

__global__ void get_wrong_index_saddles(){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=nMax) return;
    
    const int maxId = maximum[i];
    
    if(dec_largestSaddlesForMax[i] != largestSaddlesForMax[i]){
        if(wrong_rank_saddle[maxId] == 0){
            int pos = atomicAdd(&wrong_saddle_counter, 1);
            wrong_rank_saddle_index[pos * 2] = largestSaddlesForMax[i];
            wrong_rank_saddle_index[pos * 2 + 1] = dec_largestSaddlesForMax[i];
            
            wrong_rank_saddle[maxId] = 1;
        }
    }
    

}

__global__ void get_wrong_index_saddles_join(){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=nMin) return;
    
    const int minId = minimum[i];
   
    if(dec_smallestSaddlesForMin[i] != smallestSaddlesForMin[i]){
        if(wrong_rank_saddle_join[minId] == 0){
            // if(smallestSaddlesForMin[i] == 5477) printf("%d %d\n", minId, i);
            int pos = atomicAdd(&wrong_saddle_counter_join, 1);
            wrong_rank_saddle_join_index[pos * 2] = smallestSaddlesForMin[i];
            wrong_rank_saddle_join_index[pos * 2 + 1] = dec_smallestSaddlesForMin[i];
            wrong_rank_saddle_join[minId] = 1;
        }
    }

}

__device__ bool isCommonNeighbor(int dx, int dy, int dz) {
    
    for(int i = 0; i<14; i++){
        if(dx == directions[ 3 * i + 0] && dy == directions[3 * i + 1] && dz == directions[3 * i + 2]) return true;
    }
    return false;
}

__global__ void generateLookupTable() {

    for (int i = 0; i < 14; i++) {
        int dx = directions[ i * 3 + 0];
        int dy = directions[ i * 3 + 1];
        int dz = directions[ i * 3 + 2];
        int v3_count = 0;
        for (int j = 0; j < 14; j++) {
            int dx1 = directions[ j * 3 + 0];
            int dy1 = directions[ j * 3 + 1];
            int dz1 = directions[ j * 3 + 2];
            
            int dx2 = dx + dx1;
            int dy2 = dy + dy1;
            int dz2 = dz + dz1;
            // if(i == 0 && j == 10) printf("%d %d %d %d %d %d\n", )
            if(isCommonNeighbor(dx2, dy2, dz2)){
                lookupTable[i][v3_count][0] = dx2;
                lookupTable[i][v3_count][1] = dy2;
                lookupTable[i][v3_count][2] = dz2;
                v3_count++;
            }
            // else{
            //     lookupTable[i][j][0] = -2;
            //     lookupTable[i][j][1] = -2;
            //     lookupTable[i][j][2] = -2;
            // }
            
        }   
        
        while (v3_count < MAX_V3) {
            lookupTable[i][v3_count][0] = -2;
            lookupTable[i][v3_count][1] = -2;
            lookupTable[i][v3_count][2] = -2;
            v3_count++;
        }

    }
}

__global__ void fix_wrong_index_max(int direction = 0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if((i>=wrong_max_counter && direction == 0) || (i>=wrong_max_counter_2 && direction == 1)) return;
    


    if(direction == 0){
        int true_index = wrong_rank_max_index[i*2];
        int false_index = wrong_rank_max_index[i*2+1];
        
        float tmp_delta;
        float tmp_true_value = decp_data[true_index];
        float tmp_false_value = decp_data[false_index];
        // if(i == 0 && direction == 0) printf("%d: %.17f %.17f, %d: %.17f %.17f\n", true_index, decp_data[true_index], input_data[true_index], false_index, decp_data[false_index], input_data[false_index]);
        tmp_false_value = (input_data[false_index] - bound + decp_data[false_index]) / 2.0;
        int c = delta_counter[false_index] + 1;
        float d = -bound / (1 << 2);
        // float d = tmp_false_value - decp_data[false_index];
        
        float oldValue = d_deltaBuffer[false_index];
        
        if (d > oldValue) {
            swap(false_index, d);
        }  
    }

    else if(direction == 1){
        int true_index = wrong_rank_max_index_2[i*2];
        int false_index = wrong_rank_max_index_2[i*2+1];
        // if not simplified
        float tmp_delta;
        float tmp_true_value = decp_data[true_index];
        float tmp_false_value = decp_data[false_index];
        
        tmp_true_value = (input_data[true_index] - bound + decp_data[true_index]) / 2.0;
        int c = delta_counter[true_index] + 1;
        float d = -bound / (1 << 2);
        // float d = tmp_true_value - decp_data[true_index];
        
        float oldValue = d_deltaBuffer[true_index];
        
        if (d > oldValue) {
            swap(true_index, d);
        } 
    }
    

    return;
}

__global__ void fix_wrong_index_min(int direction = 0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if((i>=wrong_min_counter && direction == 0) || (i>=wrong_min_counter_2 && direction == 1)) return;
    

    if(direction == 0){
        int true_index = wrong_rank_min_index[i*2];
        int false_index = wrong_rank_min_index[i*2+1];
        
        float tmp_true_value = decp_data[true_index];
        float tmp_false_value = decp_data[false_index];
        
        tmp_true_value = (input_data[true_index] - bound + decp_data[true_index]) / 2.0;
        // float d = tmp_true_value - decp_data[true_index];
        int c = delta_counter[true_index] + 1;
        float d = -bound / (1 << 2);
        float oldValue = d_deltaBuffer[true_index];
        
        if (d > oldValue) {
            swap(true_index, d);
        }  
    }

    else if(direction == 1){
        int true_index = wrong_rank_min_index_2[i*2];
        int false_index = wrong_rank_min_index_2[i*2+1];
        
        float tmp_true_value = decp_data[true_index];
        float tmp_false_value = decp_data[false_index];
        tmp_false_value = (input_data[false_index] - bound + decp_data[false_index]) / 2.0;
        // float d = tmp_false_value - decp_data[false_index];
        int c = delta_counter[false_index] + 1;
        float d = -bound / (1 << 2);
        // printf("%d %d %.17f\n", false_index, true_index, d);
        float oldValue = d_deltaBuffer[false_index];
        
        if (d > oldValue) {
            swap(false_index, d);
        } 
    }

    return;
}
    
__global__ void fix_wrong_index_saddle(int direction = 0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if((i>=wrong_saddle_counter && direction == 0) || (i>=wrong_saddle_counter_join && direction == 1) ) return;

    
    if(direction == 0){
        int true_index = wrong_rank_saddle_index[i*2];
        int false_index = wrong_rank_saddle_index[i*2+1];
       
        float tmp_true_value = decp_data[true_index];
        float tmp_false_value = decp_data[false_index];
        if(tmp_true_value > tmp_false_value || (tmp_true_value == tmp_false_value && true_index > false_index)) return;
        int c = delta_counter[false_index] + 1;
        float d = -bound / (1 << 2);
        // float d = (input_data[false_index] - bound + decp_data[false_index])/2.0 - decp_data[false_index];
        float oldValue = d_deltaBuffer[false_index];
        if (d > oldValue) {
            swap(false_index, d);
        }  
    }

    else{
        int true_index = wrong_rank_saddle_join_index[i*2];
        int false_index = wrong_rank_saddle_join_index[i*2+1];

        float tmp_true_value = decp_data[true_index];
        float tmp_false_value = decp_data[false_index];
        // if(true_index == 5477) printf("%d: %.17f %.17f, %d: %.17f %.17f\n", true_index, decp_data[true_index], input_data[true_index], false_index, decp_data[false_index], input_data[false_index]);
        if(tmp_true_value < tmp_false_value || (tmp_true_value == tmp_false_value && true_index < false_index)) return;
        int c = delta_counter[true_index] + 1;
        float d = -bound / (1 << 2);
        // float d = (input_data[true_index] - bound + decp_data[true_index])/2.0 - decp_data[true_index];
        float oldValue = d_deltaBuffer[true_index];
        if (d > oldValue) {
            swap(true_index, d);
        }  
    }
    

    return;
}

void s_loops(dim3 gridSize, dim3 blockSize, dim3 gridSize_2saddle){
        cudaMemcpyToSymbol(wrong_max_counter, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(wrong_max_counter_2, &initialValue, sizeof(int));
        
        init_buffer<<<gridSize, blockSize>>>();
        get_wrong_index_max<<<gridSize_2saddle, blockSize>>>();
        init_delta<<<gridSize, blockSize>>>();

        int host_wrong_max_counter = 0, host_wrong_max_counter_2 = 0;
        cudaMemcpyFromSymbol(&host_wrong_max_counter, wrong_max_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_wrong_max_counter_2, wrong_max_counter_2, sizeof(int), 0, cudaMemcpyDeviceToHost);

        dim3 gridSize_wrong_max((host_wrong_max_counter + blockSize.x - 1) / blockSize.x);
        dim3 gridSize_wrong_max_2((host_wrong_max_counter_2 + blockSize.x - 1) / blockSize.x);
        
        if(host_wrong_max_counter>0){
            fix_wrong_index_max<<<gridSize_wrong_max, blockSize>>>(0);
        }

        if(host_wrong_max_counter_2>0) fix_wrong_index_max<<<gridSize_wrong_max_2, blockSize>>>(1);
        applyDeltaBuffer<<<gridSize, blockSize>>>();
}

void s_loops_join(dim3 gridSize, dim3 blockSize, dim3 gridSize_1saddle){

    cudaMemcpyToSymbol(wrong_min_counter, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(wrong_min_counter_2, &initialValue, sizeof(int));
    init_buffer<<<gridSize, blockSize>>>(1);
    get_wrong_index_min<<<gridSize_1saddle, blockSize>>>();
    
    init_delta<<<gridSize, blockSize>>>();
    
    int host_wrong_min_counter = 0, host_wrong_min_counter_2 = 0;
    cudaMemcpyFromSymbol(&host_wrong_min_counter, wrong_min_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_wrong_min_counter_2, wrong_min_counter_2, sizeof(int), 0, cudaMemcpyDeviceToHost);
    // std::cout<<host_wrong_min_counter<<", "<<host_wrong_min_counter_2<<std::endl;
    dim3 gridSize_wrong_min((host_wrong_min_counter + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_wrong_min_2((host_wrong_min_counter_2 + blockSize.x - 1) / blockSize.x);
    
    if(host_wrong_min_counter>0) fix_wrong_index_min<<<gridSize_wrong_min, blockSize>>>(0);

    if(host_wrong_min_counter_2>0) fix_wrong_index_min<<<gridSize_wrong_min_2, blockSize>>>(1);

    
    applyDeltaBuffer<<<gridSize, blockSize>>>();

}

__global__ void update_unique_sizes(int *saddleTriplets_t, int *unique_sizes, int nSaddle2) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nSaddle2) {
        saddleTriplets_t[index * 46 + 44] = unique_sizes[index];
    }
}

// void sortTripletsOnGPU(int *saddleTriplets_t, int nSaddle2, float *offset, int type = 0) {
//     thrust::device_ptr<int> d_ptr(saddleTriplets_t);
    
//     // 申请一个 `device_vector` 来存储 unique_size 结果
//     thrust::device_vector<int> unique_sizes(nSaddle2);

//     // 并行排序 & 去重
//     for (int index = 0; index < nSaddle2; index++) {
//         int sort_size;
//         cudaMemcpy(&sort_size, &saddleTriplets_t[index * 46 + 44], sizeof(int), cudaMemcpyDeviceToHost);

//         thrust::sort(d_ptr + index * 46, d_ptr + index * 46 + sort_size, OffsetComparator(offset));
//         auto new_end = thrust::unique(d_ptr + index * 46, d_ptr + index * 46 + sort_size);
        
//         // 计算 unique_size
//         unique_sizes[index] = thrust::distance(d_ptr + index * 46, new_end);
//     }

//     // 用 CUDA 内核更新 `saddleTriplets_t[index * 46 + 44]`
//     int *d_unique_sizes = thrust::raw_pointer_cast(unique_sizes.data());
//     update_unique_sizes<<<(nSaddle2 + 255) / 256, 256>>>(saddleTriplets_t, d_unique_sizes, nSaddle2);
// }

// void sortCP(int *minimum, int *maximum,
//             int *saddles1, int *saddles2,
//             int *saddlerank, int *saddle1rank,
//             float *offset, int nMin, int nMax, 
//             int nSaddle1, int nSaddle2, int type = 1){
//     thrust::device_ptr<int> minimum_ptr(minimum);
//     thrust::device_ptr<int> maximum_ptr(maximum);
//     thrust::device_ptr<int> saddles1_ptr(saddles1);
//     thrust::device_ptr<int> saddles2_ptr(saddles2);
//     thrust::device_ptr<int> indices1_ptr(saddle1rank);
//     thrust::device_ptr<int> indices2_ptr(saddlerank);

//     thrust::sort(minimum_ptr, minimum_ptr + nMin, OffsetComparator(offset));
//     thrust::sort(maximum_ptr, maximum_ptr + nMax, OffsetComparator(offset));
//     thrust::sort(saddles1_ptr, saddles1_ptr + nSaddle1, OffsetComparator(offset));
//     thrust::sort(saddles2_ptr, saddles2_ptr + nSaddle2, OffsetComparator(offset));

//     thrust::sequence(indices1_ptr, indices1_ptr + nSaddle1);
//     thrust::sequence(indices2_ptr, indices2_ptr + nSaddle2);


//     // thrust::sort_by_key(saddles1_ptr, saddles1_ptr + nSaddle1, indices1_ptr, OffsetComparator(offset));
//     // thrust::sort_by_key(saddles2_ptr, saddles2_ptr + nSaddle2, indices2_ptr, OffsetComparator(offset));

// }

void saveArrayToBin(const float* arr, size_t size, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    outFile.write(reinterpret_cast<const char*>(arr), size * sizeof(float));

    outFile.close();
}

__device__ void c_loop_local(int index, int direction = 0){      
    // preservation of split tree?->decrease f
    
    if (direction == 0){
        
        // if vertex is a regular point.
        int x = index % width;
        int y = (index / width) % height;
        int z = (index / (width * height)) % depth;
        float input_value = input_data[index];
        float decp_value = decp_data[index];
        if (vertex_type[index]!=4){
            int c = delta_counter[index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((input_value - bound) + decp_value) / 2.0 - decp_value;
            float oldValue = d_deltaBuffer[index];
            
            if (d > oldValue) {
                swap(index, d);
            }  

            return;
        
        }
        else{
            
            // if is a maximum in the original data;
            int largest_index = index;
            float largest_value = decp_data[largest_index];
            for(int i = 0; i< maxNeighbors;i++){
                int dx = directions[3 * i];
                int dy = directions[3 * i + 1];
                int dz = directions[3 * i + 2];

                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                int neighbor = x + dx + (y + dy + (z + dz) * height) * width;
                if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth || neighbor < 0 || neighbor >= num_Elements) continue;
                
                // neighbor = adjacency[maxNeighbors * index + i];
                // if(neighbor == -1) continue;
                float neighbor_value = decp_data[neighbor];
                if(islarger_shared(neighbor, largest_index, neighbor_value, largest_value))
                {
                    largest_index = neighbor;
                    largest_value = decp_data[largest_index];
                }
            }
            
            if(decp_value>largest_value or(decp_value==largest_value and index>largest_index)){
                return;
            }
            int c = delta_counter[largest_index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((input_data[largest_index] - bound) + largest_value) / 2.0 - largest_value;
            
            float oldValue = d_deltaBuffer[largest_index];
            if (d > oldValue) {
                swap(largest_index, d);
            }  

            return;
        }
    }
    
    
    else if (direction == 1){
        int x = index % width;
        int y = (index / width) % height;
        int z = (index / (width * height)) % depth;
        float input_value = input_data[index];
        float decp_value = decp_data[index];
        if (vertex_type[index]!=0){
            int smallest_index = index;
            float smallest_value = input_data[smallest_index];
            for(int i = 0; i< maxNeighbors;i++){
                int dx = directions[3 * i];
                int dy = directions[3 * i + 1];
                int dz = directions[3 * i + 2];

                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                int neighbor = x + dx + (y + dy + (z + dz) * height) * width;
                if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth || neighbor<0 || neighbor >= num_Elements) continue;
                
                // neighbor = adjacency[maxNeighbors * index + i];
                // if(neighbor == -1) continue;
                float neighbor_value = input_data[neighbor];
                if(isless_shared(neighbor, smallest_index, neighbor_value, smallest_value)){
                    smallest_index = neighbor;
                    smallest_value = input_data[smallest_index];
                }
            }
            // for(int i = 0; i<maxNeighbors;i++){
            //     int neighbor = adjacency[maxNeighbors * index + i];
            //     if(neighbor == -1) continue;
            //     if(isless(neighbor, smallest_index, input_data))
            //     {
            //         smallest_index = neighbor;
            //     }
            // }
            float decp_smallest_value = decp_data[smallest_index];
            int c = delta_counter[smallest_index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((smallest_value - bound) + decp_smallest_value) / 2.0 - decp_smallest_value;
            if(decp_value>decp_smallest_value or (decp_value==decp_smallest_value and index>smallest_index)){
                return;
            }
            float oldValue = d_deltaBuffer[smallest_index];
            if (d > oldValue) {
                swap(smallest_index, d);
            }  
            return;
        
        }
    
        else{
            int c = delta_counter[index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((input_value - bound) + decp_value) / 2.0 - decp_value;
            float oldValue = d_deltaBuffer[index];
            if (d > oldValue) {
                swap(index, d);
            } 
            return;
        }
    }   

    else{
        fix_saddle(index);
    }

    
    return;
}

__global__ void classifyVertex_CUDA(int type = 0, int local = 1) {
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= num_Elements ) return;
    // 

    float* heightMap = input_data;
    int* results = vertex_type;
    int* lowerStars_t = lowerStars;
    int* upperStars_t = upperStars;

    if(type == 1){
        heightMap = decp_data;
        results = dec_vertex_type;
        lowerStars_t = dec_lowerStars;
        upperStars_t = dec_upperStars;
    }

    __shared__ float smem[TILE_SIZE + 2][TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x + 1;  // 1-based index in shared memory (考虑Halo)
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    int gx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int gy = blockIdx.y * TILE_SIZE+ threadIdx.y;
    int gz = blockIdx.z * TILE_SIZE + threadIdx.z;
    
    // // 读取当前 voxel
    if (gx < width && gy < height && gz < depth) {
        smem[tx][ty][tz] = heightMap[gz * width * height + gy * width + gx];
    }

    
    if( threadIdx.x == 0 || threadIdx.x == TILE_SIZE - 1 ||
        threadIdx.y == 0 || threadIdx.y == TILE_SIZE - 1  ||
        threadIdx.z == 0 || threadIdx.z == TILE_SIZE - 1 ){
        for (int i = 0; i < 14; i++) {
            int dx = neighborOffsets[i][0];
            int dy = neighborOffsets[i][1];
            int dz = neighborOffsets[i][2];

            int nx = gx + dx;
            int ny = gy + dy;
            int nz = gz + dz;

            int ntx = tx + dx;
            int nty = ty + dy;
            int ntz = tz + dz;

            // 只有 block 边界的线程负责加载 Halo Cells
            if(nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz <depth){
                smem[ntx][nty][ntz] = heightMap[nz * height * width + ny * width + nx];
            }
        }
    }


    __syncthreads();  // 确保所有 shared memory 都加载完毕
    
    int g_idx = gz * width * height + gy * width + gx;
    if(g_idx >= num_Elements) return;
    if (!(gx < width && gy < height && gz < depth))  return;
    if(updated_vertex[g_idx] != 0 && local == 1) return;

    float currentHeight = smem[tx][ty][tz];
    // float currentHeight = heightMap[g_idx];
    int vertexId = g_idx;

    int lowerCount = 0, upperCount = 0;
    int lowerStar[MAX_NEIGHBORS] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int upperStar[MAX_NEIGHBORS]= {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    
    int neighbor_size = 0;
    int binary[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int vertex_binary[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int d = 0; d < maxNeighbors; d++) {
        
        int dirX = directions[d * 3];     
        int dirY = directions[d * 3 + 1]; 
        int dirZ = directions[d * 3 + 2]; 
        int newX = gx + dirX;
        int newY = gy + dirY;
        int newZ = gz + dirZ;
        int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
        
        if (newX >= 0 && newX < width && newY >= 0 && newY < height && r >= 0 && r < width*height*depth && newZ<depth && newZ>=0) {
    
            int smem_x = tx + (newX - gx);
            int smem_y = ty + (newY - gy);
            int smem_z = tz + (newZ - gz);
            
            
            float neighbor_value = smem[smem_x][smem_y][smem_z];

            if (neighbor_value < currentHeight || (neighbor_value == currentHeight && r < g_idx)) {
                if(type==0) lowerStars_t[vertexId*(maxNeighbors+1) + lowerCount] = r;
                binary[maxNeighbors - 1 - neighbor_size] = 0;
                lowerStar[lowerCount] = r;
                lowerCount++;
                
            } else if (neighbor_value > currentHeight || (neighbor_value == currentHeight && r > g_idx)) {
                if(type==0) upperStars_t[vertexId*(maxNeighbors+1) + upperCount] = r;
                upperStar[upperCount] = r;
                binary[maxNeighbors - 1 - neighbor_size] = 1;
                upperCount++;
                
            }
            vertex_binary[13 - d] = 1;
            neighbor_size++;
        }
    }

    if(type==0) lowerStars_t[vertexId*(maxNeighbors+1) + maxNeighbors] = lowerCount;
    if(type==0) upperStars_t[vertexId*(maxNeighbors+1) + maxNeighbors] = upperCount;
    
    int decimal_value = 0;
    for (int i = 0; i < maxNeighbors; i++) {
        decimal_value = (decimal_value << 1) | binary[i]; // 左移并加上当前位
    }

    int vertex_types = 0;
    for (int i = 0; i < maxNeighbors; i++) {
        vertex_types= (vertex_types << 1) | vertex_binary[i]; // 左移并加上当前位
    }
    int keyIndex = binarySearchLUT(keys, NUM_KEYS, vertex_types);
    
    int LUT_result = 5;
    if(upperCount == 0) LUT_result = 4;
    else if(lowerCount == 0) LUT_result = 0;
    else if (keyIndex != -1) {
        LUT_result = LUT_cuda[lutOffsets[keyIndex] + decimal_value];
    } else {
        LUT_result = -1; // 未找到
    }

    
    if(type==0){
        results[g_idx] = LUT_result;
        
    }
    else{

        // if(host_count_f_min>0){
        //     dim3 gridSize_fmin((host_count_f_min + blockSize.x - 1) / blockSize.x);
        //     c_loop<<<gridSize_fmin, blockSize>>>(1);
        //     cudaDeviceSynchronize();
        // }
                
        // if(host_count_f_saddle>0 ){
        //     dim3 gridSize_saddle((host_count_f_saddle + blockSize.x - 1) / blockSize.x);
        //     c_loop<<<gridSize_saddle, blockSize>>>(2);
        //     cudaDeviceSynchronize();
        // }
        
        // if(host_count_f_max>0){
        //     dim3 gridSize_fmax((host_count_f_max + blockSize.x - 1) / blockSize.x);
        //     c_loop<<<gridSize_fmax, blockSize>>>(0);
        //     cudaDeviceSynchronize();
        // }
        int ori_type = vertex_type[g_idx];
        if(LUT_result != ori_type){
                // maximum
                if((LUT_result==4 and ori_type!=4) or (LUT_result!=4 and ori_type==4)){
                    int idx_fp_max = atomicAdd(&count_f_max, 1);
                    all_max[idx_fp_max] = g_idx;
                    // c_loop_local(g_idx, 0);
                    
                } 
                if((LUT_result==0 and ori_type!=0) or (LUT_result!=0 and ori_type==0)){
                    int idx_fp_min = atomicAdd(&count_f_min, 1);
                    all_min[idx_fp_min] = g_idx;
                    // c_loop_local(g_idx, 1);
                }
                if((LUT_result==2 and ori_type!=2) or (LUT_result!=2 and ori_type==2)){
                    int idx_fp_saddle = atomicAdd(&count_f_saddle, 1);
                    all_saddle[idx_fp_saddle] = g_idx;
                    // c_loop_local(g_idx, 2);
                    // fix_saddle_local(g_idx, lowerStar, lowerCount, lowerStars+g_idx*(maxNeighbors + 1), lowerStars[g_idx*(maxNeighbors + 1) +maxNeighbors]);
                }

                if((LUT_result==1 and ori_type!=1) or (LUT_result!=1 and ori_type==1)){
                    int idx_fp_saddle = atomicAdd(&count_f_saddle, 1);
                    all_saddle[idx_fp_saddle] = g_idx;
                    // c_loop_local(g_idx, 2);
                    // fix_saddle_local(g_idx, lowerStar, lowerCount, lowerStars+g_idx*(maxNeighbors + 1), lowerStars[g_idx*(maxNeighbors + 1) +maxNeighbors]);
                }

                if((LUT_result==3 and ori_type!=3) or (LUT_result!=3 and ori_type==3)){
                    int idx_fp_saddle = atomicAdd(&count_f_saddle, 1);
                    all_saddle[idx_fp_saddle] = g_idx;
                    // c_loop_local(g_idx, 2);
                    // fix_saddle_local(g_idx, lowerStar, lowerCount, lowerStars+g_idx*(maxNeighbors + 1), lowerStars[g_idx*(maxNeighbors + 1) +maxNeighbors]);
                }


        }

        else if(LUT_result==2 and ori_type==2 || LUT_result== 1 and ori_type==1 || LUT_result== 3 and ori_type==3){
                
            if(areVectorsDifferent_local(g_idx, lowerStar, lowerCount)){
                int idx_fp_saddle = atomicAdd(&count_f_saddle, 1);
                all_saddle[idx_fp_saddle] = g_idx;
                // fix_saddle_local(g_idx, lowerStar, lowerCount, lowerStars+g_idx*(maxNeighbors + 1), lowerStars[g_idx*(maxNeighbors + 1) +maxNeighbors]);
            }
        }
    }
}

void c_loops(dim3 gridSize, dim3 blockSize, int host_count_f_max, int host_count_f_min, int host_count_f_saddle){
    float vertexClassification = 0.0;
    float get_fcp = 0.0;
    float c_loopt = 0.0;
    float init_deltat = 0.0;
    float apply_deltat = 0.0;

    
    dim3 blockDim1(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 gridDim1((width_host + TILE_SIZE - 1) / TILE_SIZE,
                 (height_host + TILE_SIZE - 1) / TILE_SIZE,
                 (depth_host + TILE_SIZE - 1) / TILE_SIZE);

    cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_saddle, &initialValue, sizeof(int));
    auto start = std::chrono::high_resolution_clock::now();

    init_delta<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    classifyVertex_CUDA<<<gridDim1, blockDim1>>>(1, 0);
    cudaDeviceSynchronize();
    // init_update<<<gridSize, blockSize>>>();
    // cudaDeviceSynchronize();
    // applyDeltaBuffer<<<gridSize, blockSize>>>();
    // cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    vertexClassification += duration.count();
    

    // cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
    // cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
    // cudaMemcpyToSymbol(count_f_saddle, &initialValue, sizeof(int));
    start = std::chrono::high_resolution_clock::now();
    // get_false_criticle_points<<<gridSize, blockSize>>>(); 
    // cudaDeviceSynchronize();   
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    get_fcp += duration.count();

    cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_saddle, count_f_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
    int cnt = 0;
    while(host_count_f_max>0 or host_count_f_min>0 or host_count_f_saddle>0){
        // std::cout<<"c_loops: "<<host_count_f_max<<", "<<host_count_f_min<<", "<<host_count_f_saddle<<std::endl;
        cnt++;
        start = std::chrono::high_resolution_clock::now();

        init_delta<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        init_deltat += duration.count();


        start = std::chrono::high_resolution_clock::now();
        if(host_count_f_min>0){
            dim3 gridSize_fmin((host_count_f_min + blockSize.x - 1) / blockSize.x);
            c_loop<<<gridSize_fmin, blockSize>>>(1);
            cudaDeviceSynchronize();
        }
                
        if(host_count_f_saddle>0 ){
            dim3 gridSize_saddle((host_count_f_saddle + blockSize.x - 1) / blockSize.x);
            c_loop<<<gridSize_saddle, blockSize>>>(2);
            cudaDeviceSynchronize();
        }
        
        if(host_count_f_max>0){
            dim3 gridSize_fmax((host_count_f_max + blockSize.x - 1) / blockSize.x);
            c_loop<<<gridSize_fmax, blockSize>>>(0);
            cudaDeviceSynchronize();
        }
            
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        c_loopt += duration.count();

        start = std::chrono::high_resolution_clock::now();

        init_update<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();
        applyDeltaBuffer<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();
        
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        apply_deltat += duration.count();

        cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_saddle, &initialValue, sizeof(int));

        start = std::chrono::high_resolution_clock::now();
        
        // init_delta<<<gridSize, blockSize>>>();
        // cudaDeviceSynchronize();
        classifyVertex_CUDA<<<gridDim1, blockDim1>>>(1);
        cudaDeviceSynchronize();
        // applyDeltaBuffer<<<gridSize, blockSize>>>();
        // cudaDeviceSynchronize();
        // init_update<<<gridSize, blockSize>>>();
        // cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        vertexClassification += duration.count();
        

        

        start = std::chrono::high_resolution_clock::now();
        // get_false_criticle_points<<<gridSize, blockSize>>>();    
        // cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        get_fcp += duration.count();

        cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_saddle, count_f_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
    }
    float w_time = vertexClassification + get_fcp + c_loopt + init_deltat + apply_deltat;
    // std::cout<<"number of iteration: "<<cnt<<", whole time: "<<w_time<<" vertexClassification: "<<vertexClassification/1000<<", "<<"get_fcp: "<<get_fcp/1000<<"c_loopt: "<<c_loopt/1000<<"init_deltat: "<<init_deltat/1000<<"apply_deltat: "<<apply_deltat/1000;
}

__global__ void fixpath(int direction = 0){
        float delta;
        int true_index = -1;
        int false_index = -1;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if((index>=num_false_cases && direction == 0) || (index>=num_false_cases1 && direction == 1)) return;
        
        if(direction == 0){
            const int i = wrong_neighbors_index[index];
            int current_id = i;
            
            int largest_neighbor = current_id;  
            while (true) {
                int next_largest_neighbor = largest_neighbor;
                int dec_next_largest_neighbor = largest_neighbor;
                int x = current_id % width;
                int y = (current_id / width) % height;
                int z = (current_id / (width * height)) % depth;
                for(int i = 0; i< maxNeighbors;i++){
                    int dx = directions[3 * i];
                    int dy = directions[3 * i + 1];
                    int dz = directions[3 * i + 2];

                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    int neighbor_id = x + dx + (y + dy + (z + dz) * height) * width;
                    if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth || neighbor_id <  0 || neighbor_id >= num_Elements) continue;
                    
                    // int neighbor = adjacency[maxNeighbors * index + i];
                    // if(neighbor == -1) continue;
                    if (input_data[next_largest_neighbor] < input_data[neighbor_id] || 
                    (input_data[next_largest_neighbor] == input_data[neighbor_id] && next_largest_neighbor < neighbor_id)) {
                        next_largest_neighbor = neighbor_id;
                    }
                    if (decp_data[dec_next_largest_neighbor] < decp_data[neighbor_id] || 
                    (decp_data[dec_next_largest_neighbor] == decp_data[neighbor_id] && dec_next_largest_neighbor < neighbor_id)) {
                        dec_next_largest_neighbor = neighbor_id;
                    }
                }
                // for (int j = 0; j < maxNeighbors; j++) {
                //     int neighbor_id = adjacency[maxNeighbors * current_id + j];
                //     if (neighbor_id == -1) continue;  
                    
                //     if (input_data[next_largest_neighbor] < input_data[neighbor_id] || 
                //     (input_data[next_largest_neighbor] == input_data[neighbor_id] && next_largest_neighbor < neighbor_id)) {
                //         next_largest_neighbor = neighbor_id;
                //     }
                //     if (decp_data[dec_next_largest_neighbor] < decp_data[neighbor_id] || 
                //     (decp_data[dec_next_largest_neighbor] == decp_data[neighbor_id] && dec_next_largest_neighbor < neighbor_id)) {
                //         dec_next_largest_neighbor = neighbor_id;
                //     }
                // }

                if (next_largest_neighbor != dec_next_largest_neighbor) {
                    true_index = next_largest_neighbor;
                    false_index = dec_next_largest_neighbor;
                    break;
                }
                if (next_largest_neighbor == largest_neighbor) break;

                current_id = next_largest_neighbor;
                largest_neighbor = next_largest_neighbor;
            }
            
            
            if(false_index==true_index){
                
                return;
            }
            
            int c = delta_counter[false_index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((input_data[false_index] - bound) + decp_data[false_index]) / 2.0 - decp_data[false_index];
        
            float oldValue = d_deltaBuffer[false_index];
            if (d > oldValue) {
                swap(false_index, d);
            } 
            
             

            return;
            
        }

        else {
            const int i = wrong_neighbors_ds_index[index];
            int current_id = i;
            
            int largest_neighbor = current_id;  
            
            while (true) {
                int next_largest_neighbor = largest_neighbor;
                int dec_next_largest_neighbor = largest_neighbor;
                
                // for (int j = 0; j < maxNeighbors; j++) {
                //     int neighbor_id = adjacency[maxNeighbors * current_id + j];
                //     if (neighbor_id == -1) continue;  
                    
                //     if (input_data[next_largest_neighbor] > input_data[neighbor_id] || 
                //     (input_data[next_largest_neighbor] == input_data[neighbor_id] && next_largest_neighbor > neighbor_id)) {
                //         next_largest_neighbor = neighbor_id;
                //     }
                //     if (decp_data[dec_next_largest_neighbor] > decp_data[neighbor_id] || 
                //     (decp_data[dec_next_largest_neighbor] == decp_data[neighbor_id] && dec_next_largest_neighbor > neighbor_id)) {
                //         dec_next_largest_neighbor = neighbor_id;
                //     }
                // }
                int x = current_id % width;
                int y = (current_id / width) % height;
                int z = (current_id / (width * height)) % depth;
                for(int i = 0; i< maxNeighbors;i++){
                    int dx = directions[3 * i];
                    int dy = directions[3 * i + 1];
                    int dz = directions[3 * i + 2];

                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    int neighbor_id = x + dx + (y + dy + (z + dz) * height) * width;
                    if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth || neighbor_id <  0 || neighbor_id >= num_Elements) continue;
                        
                    // int neighbor = adjacency[maxNeighbors * index + i];
                    // if(neighbor == -1) continue;
                    if (input_data[next_largest_neighbor] > input_data[neighbor_id] || 
                    (input_data[next_largest_neighbor] == input_data[neighbor_id] && next_largest_neighbor > neighbor_id)) {
                        next_largest_neighbor = neighbor_id;
                    }
                    if (decp_data[dec_next_largest_neighbor] > decp_data[neighbor_id] || 
                    (decp_data[dec_next_largest_neighbor] == decp_data[neighbor_id] && dec_next_largest_neighbor > neighbor_id)) {
                        dec_next_largest_neighbor = neighbor_id;
                    }
                }
                if (next_largest_neighbor != dec_next_largest_neighbor) 
                {
                    true_index = next_largest_neighbor;
                    false_index = dec_next_largest_neighbor;
                    break;
                }
                if (next_largest_neighbor == largest_neighbor) break;

                current_id = next_largest_neighbor;
                largest_neighbor = next_largest_neighbor;
            }

            if(false_index==true_index) return;

            int c = delta_counter[true_index] + 1;
            float d = -bound / (1 << 2);
            // float d = ((input_data[true_index] - bound) + decp_data[true_index]) / 2.0 - decp_data[true_index];
            float oldValue = d_deltaBuffer[true_index];
            if (d > oldValue) {
                swap(true_index, d);
            } 
            return;
            
        
        }
        return;
    }


void r_loops(dim3 gridSize, dim3 blockSize, int host_number_of_false_cases, int host_number_of_false_cases1){
        
    init_delta<<<gridSize, blockSize>>>();
    if(host_number_of_false_cases>0){

        dim3 gridSize_wrong_case((host_number_of_false_cases + blockSize.x - 1) / blockSize.x);
        fixpath<<<gridSize_wrong_case, blockSize>>>(0);
    
    }
    

    if(host_number_of_false_cases1>0){
        dim3 gridSize_wrong_case1((host_number_of_false_cases1 + blockSize.x - 1) / blockSize.x);
        fixpath<<<gridSize_wrong_case1, blockSize>>>(1);
    
    }
    applyDeltaBuffer<<<gridSize, blockSize>>>();
}

void saddle_loops(dim3 gridSize_max, dim3 gridSize, dim3 blockSize){
    cudaMemcpyToSymbol(wrong_saddle_counter, &initialValue, sizeof(int));
    init_saddle_rank_buffer<<<gridSize, blockSize>>>();
    // get_wrong_index_saddles<<<gridSize_max, blockSize>>>();
    init_delta<<<gridSize, blockSize>>>();
    // std::cout<<wrong_saddle_counter<<std::endl;
    int host_wrong_saddle_counter = 0;
    cudaMemcpyFromSymbol(&host_wrong_saddle_counter, wrong_saddle_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    dim3 gridSize_wrong_saddle((host_wrong_saddle_counter + blockSize.x - 1) / blockSize.x);

    if(host_wrong_saddle_counter>0) fix_wrong_index_saddle<<<gridSize_wrong_saddle, blockSize>>>(0);
    applyDeltaBuffer<<<gridSize, blockSize>>>();
    
}

void saddle_loops_join(dim3 gridSize_min, dim3 gridSize, dim3 blockSize){
    cudaMemcpyToSymbol(wrong_saddle_counter_join, &initialValue, sizeof(int));
    init_saddle_rank_buffer<<<gridSize, blockSize>>>(1);
    // get_wrong_index_saddles_join<<<gridSize_min, blockSize>>>();
    init_delta<<<gridSize, blockSize>>>();
    int host_wrong_saddle_counter_join = 0;
    cudaMemcpyFromSymbol(&host_wrong_saddle_counter_join, wrong_saddle_counter_join, sizeof(int), 0, cudaMemcpyDeviceToHost);
    dim3 gridSize_wrong_saddle_join((host_wrong_saddle_counter_join + blockSize.x - 1) / blockSize.x);

    if(host_wrong_saddle_counter_join >0){
        dim3 gridSize_wrong_saddle_join((host_wrong_saddle_counter_join + blockSize.x - 1) / blockSize.x);
        fix_wrong_index_saddle<<<gridSize_wrong_saddle_join, blockSize>>>(1);
        applyDeltaBuffer<<<gridSize, blockSize>>>();
    }
}

float calculateMSE(const float* original, const float* compressed) {


    float mse = 0.0;
    for (size_t i = 0; i < num_Elements_host; i++) {
        mse += std::pow(static_cast<float>(original[i]) - compressed[i], 2);
    }
    mse /= num_Elements_host;
    return mse;
}

float calculatePSNR( float* original, float* compressed, float maxValue) {
    float mse = calculateMSE(original, compressed);
    if (mse == 0) {
        return std::numeric_limits<float>::infinity(); // Perfect match
    }
    float psnr = -20.0*log10(sqrt(mse)/maxValue);
    return psnr;
}

std::vector<uint8_t> encodeTo3BitBitmap(const std::vector<int>& data) {
    std::vector<uint8_t> bitmap; // 存储结果的位图
    uint8_t currentByte = 0; // 当前字节
    int bitIndex = 0; // 当前位的索引

    for (int value : data) {
        if (value < 0 || value > 7) {
            
            std::cerr << "错误：输入值超出范围 (0-6)。"<<value << std::endl;
        return {};
        }

        // 将当前值的3位插入到当前字节
        currentByte |= (value << (bitIndex)); // 将值左移到正确的位置
        bitIndex += 3; // 增加3位

        // 检查是否需要写入字节
        if (bitIndex >= 8) { // 如果当前字节已填满（8位）
            bitmap.push_back(currentByte); // 将填满的字节加入位图
            bitIndex -= 8; // 减去已填满的位
            currentByte = (value >> (3 - bitIndex)); // 如果有剩余的位，存储到下一个字节的起始位置
        }
    }

    // 如果最后有未完成的字节，写入它
    if (bitIndex > 0) {
        bitmap.push_back(currentByte);
    }

    return bitmap;
}

void cost(std::string filename, float* decp_data, float* decp_data_copy, float* input_data, std::string compressor_id, std::vector<int> delta_counter){
        std::vector<int> indexs(num_Elements_host, 0);
        std::vector<float> edits;
        std::vector<unsigned long long> exponents;
        std::vector<unsigned long long> mantissas;

        int cnt = 0;
        for (int i=0;i<num_Elements_host;i++){
            if (decp_data_copy[i]!=decp_data[i]){
                indexs[i] = delta_counter[i];
                
                if(indexs[i]==7){
                    // edits.push_back(-(decp_data_copy[i] - (input_data[i] - host_bound)));
                    edits.push_back(input_data[i]);
                    // edits.push_back((input_data[i] - host_bound));
                }
                cnt++;
            }
        }
        
        
        std::vector<uint8_t> bitmap = encodeTo3BitBitmap(indexs);

        
        std::string indexfilename = "/pscratch/sd/y/yuxiaoli/MSCz/data1"+filename+std::to_string(host_bound)+".bin";
        std::string editsfilename = "/pscratch/sd/y/yuxiaoli/MSCz/data_edits"+filename+std::to_string(host_bound)+".bin";
        std::string compressedindex = "/pscratch/sd/y/yuxiaoli/MSCz/data1"+filename+std::to_string(host_bound)+".bin.zst";
        std::string compressededits = "/pscratch/sd/y/yuxiaoli/MSCz/data_edits"+filename+std::to_string(host_bound)+".bin.zst";
        
        // Shockwave, 64, 64, 512

        float ratio = float(cnt)/(num_Elements_host);
        std::cout<<cnt<<","<<ratio<<std::endl;

        std::ofstream file(indexfilename, std::ios::binary | std::ios::out);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(indexs.data()), indexs.size());
            file.close();
        } else {
            std::cerr << "cannot open file: " << filename << " ." << std::endl;
        }
        
        std::string command;
        command = "zstd -f " + indexfilename;
        std::cout << "Executing command: " << command << std::endl;
        int result = std::system(command.c_str());
        if (result == 0) {
            
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }

        std::ofstream file1(editsfilename, std::ios::binary | std::ios::out);
        if (file1.is_open()) {
            file1.write(reinterpret_cast<const char*>(edits.data()), edits.size()*sizeof(float));
            file1.close();
        } else {
            std::cerr << "cannot open file: " << filename << " ." << std::endl;
        }
        
        
        command = "zstd -f " + editsfilename;
        std::cout << "Executing command: " << command << std::endl;
        result = std::system(command.c_str());
        if (result == 0) {
            
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
       

    
        
        std::uintmax_t compressed_indexSize = std::filesystem::file_size(compressedindex);
        std::uintmax_t compressed_editSize =std::filesystem::file_size(compressededits);
        std::uintmax_t original_indexSize = std::filesystem::file_size(indexfilename);
        std::uintmax_t original_editSize = std::filesystem::file_size(editsfilename);
        std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
        std::uintmax_t compressed_dataSize = cmpSize;
        std::cout<<"compressed datasize:"<<cmpSize<<std::endl;

        float overall_ratio = float(original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
    
        float bitRate = 64/overall_ratio; 

        float psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
        float fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);

        std::ofstream outFile3("../stat_result/result_"+filename+"_"+compressor_id+"_detailed_additional_time.txt", std::ios::app);

        
        if (!outFile3) {
            std::cerr << "Unable to open file for writing." << std::endl;
            return; // 返回错误码
        }

        
        outFile3 << std::to_string(host_bound)<<":" << std::endl;
        outFile3 << std::setprecision(17)<< "related_error: "<<er << std::endl;
        outFile3 << std::setprecision(17)<< "absolute_error: "<<host_bound << std::endl;
        outFile3 << std::setprecision(17)<< "OCR: "<<overall_ratio << std::endl;
        outFile3 <<std::setprecision(17)<< "CR: "<<float(original_dataSize)/compressed_dataSize << std::endl;
        outFile3 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
        outFile3 << std::setprecision(17)<<"BR: "<< 64/(float(original_indexSize)/compressed_dataSize) << std::endl;
        outFile3 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
        outFile3 << std::setprecision(17)<<"fixed_psnr: "<<fixed_psnr << std::endl;
        

        
        outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
        // outFile3 << std::setprecision(17)<<"compression_time: "<<compression_time<< std::endl;
        // outFile3 << std::setprecision(17)<<"additional_time: "<<additional_time<< std::endl;
        outFile3 << "\n" << std::endl;

        outFile3.close();

        std::cout << "Variables have been appended to output.txt" << std::endl;
        return;
};

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv){
    std::cout<<LUT_SIZE<<std::endl;
    
    std::cout << std::fixed << std::setprecision(16);
    std::string dimension = argv[1];
    er = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    host_sim = std::stod(argv[4]);

    
    std::istringstream iss(dimension);
    char delimiter;
    

    bool preserve_join_tree = true;
    

    if (std::getline(iss, file_path, ',')) {
        
        if (iss >> width_host >> delimiter && delimiter == ',' &&
            iss >> height_host >> delimiter && delimiter == ',' &&
            iss >> depth_host) {
            std::cout << "Filepath: " << file_path << std::endl;
            std::cout << "Width: " << width_host << std::endl;
            std::cout << "Height: " << height_host << std::endl;
            std::cout << "Depth: " << depth_host << std::endl;
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for file" << std::endl;
    }

    
    num_Elements_host = width_host*height_host*depth_host;
    int numFaces_host = maxNeighbors_host * num_Elements_host;

    
    float *input_data_host,  *decp_data_host, *decp_data_copy, *d_deltaBuffer_host;
    std::cout<<file_path<<std::endl;
    
    
    cudaMalloc(&input_data_host, num_Elements_host * sizeof(float));
    cudaMalloc(&decp_data_host, num_Elements_host * sizeof(float));
    cudaMalloc(&decp_data_copy, num_Elements_host * sizeof(float));
    cudaMalloc(&d_deltaBuffer_host, num_Elements_host * sizeof(float));
    
    
    cudaMemcpyToSymbol(d_deltaBuffer, &d_deltaBuffer_host, sizeof(float*));
    
    getdata(file_path, input_data_host, decp_data_host, decp_data_copy,
            er, host_bound, num_Elements_host);
    float elapsedTime = 0.0;
    cudaEvent_t startt, stop, start;
    cudaEventCreate(&startt);
    cudaEventCreate(&stop);
    cudaEventCreate(&start);
    cudaEventRecord(startt, 0);
    int *dec_vertex_type_host, *vertex_type_host, *vertex_cells_host, *delta_counter_host,
        *DS_M_host, *AS_M_host, *dec_DS_M_host, *dec_AS_M_host, *lowerStars_host, *dec_lowerStars_host,
        *upperStars_host, *dec_upperStars_host, *adjacency_host, *minimum_host, *dec_minimum_host, *maximum_host, *dec_maximum_host,
        *saddles1_host, *dec_saddles1_host, *saddles2_host, *dec_saddles2_host, *reachable_saddle_for_Max_host, *dec_reachable_saddle_for_Max_host, *max_index_host, 
        *reachable_saddle_for_Min_host, *dec_reachable_saddle_for_Min_host;
    
    int *or_saddle_max_map_host, *wrong_neighbors_host, *wrong_neighbors1_host, *wrong_neighbors_index_host, *wrong_rank_max_host, *wrong_rank_max_index_host, *wrong_rank_saddle_host, *wrong_rank_saddle_index_host, *wrong_rank_max_2_host, *wrong_rank_max_index_2_host;
    int *or_saddle_min_map_host, *wrong_neighbors_ds_host, *wrong_neighbors_ds_index_host, *wrong_rank_min_host, *wrong_rank_min_index_host, *wrong_rank_min_2_host, *wrong_rank_min_index_2_host;
    int *wrong_rank_saddle_join_host, *wrong_rank_saddle_join_index_host;
    int *all_min_host, *all_max_host, *all_saddle_host, *updated_vertex_host;

    

    cudaMalloc(&dec_vertex_type_host, num_Elements_host * sizeof(int));
    cudaMalloc(&updated_vertex_host, num_Elements_host * sizeof(int));
    cudaMalloc(&vertex_type_host, num_Elements_host * sizeof(int));
    cudaMalloc(&delta_counter_host, num_Elements_host * sizeof(int));
    cudaMalloc(&all_min_host, num_Elements_host * sizeof(int));
    cudaMalloc(&all_max_host, num_Elements_host * sizeof(int));
    cudaMalloc(&all_saddle_host, num_Elements_host * sizeof(int));
    cudaMalloc(&DS_M_host, num_Elements_host * sizeof(int));
    cudaMalloc(&AS_M_host, num_Elements_host * sizeof(int));
    cudaMalloc(&dec_DS_M_host, num_Elements_host * sizeof(int));
    cudaMalloc(&dec_AS_M_host, num_Elements_host * sizeof(int));
    
    cudaMalloc(&wrong_neighbors_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_neighbors_index_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_rank_max_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_rank_max_index_host, num_Elements_host * 2 * sizeof(int));
    cudaMalloc(&wrong_rank_saddle_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_rank_saddle_index_host, num_Elements_host * 2 * sizeof(int));
    cudaMalloc(&wrong_rank_max_2_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_rank_max_index_2_host, num_Elements_host * 2 * sizeof(int));
    
    cudaMalloc(&wrong_neighbors_ds_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_neighbors_ds_index_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_rank_min_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_rank_min_index_host, num_Elements_host * 2 * sizeof(int));
    cudaMalloc(&wrong_rank_saddle_join_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_rank_saddle_join_index_host, num_Elements_host * 2 * sizeof(int));
    cudaMalloc(&wrong_rank_min_2_host, num_Elements_host * sizeof(int));
    cudaMalloc(&wrong_rank_min_index_2_host, num_Elements_host * 2 * sizeof(int));
    cudaMalloc(&lowerStars_host, (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    // cudaMalloc(&dec_lowerStars_host, (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    
    cudaMalloc(&upperStars_host, (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    // cudaMalloc(&dec_upperStars_host, (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    
    // cudaMalloc(&adjacency_host, maxNeighbors_host * num_Elements_host * sizeof(int));
    cudaMalloc(&minimum_host, num_Elements_host * sizeof(int));
    // cudaMalloc(&dec_minimum_host, num_Elements_host * sizeof(int));
    cudaMalloc(&saddles1_host, num_Elements_host * sizeof(int));
    // cudaMalloc(&dec_saddles1_host, num_Elements_host * sizeof(int));
    cudaMalloc(&saddles2_host, num_Elements_host * sizeof(int));
    // cudaMalloc(&dec_saddles2_host, num_Elements_host * sizeof(int));
    cudaMalloc(&maximum_host, num_Elements_host * sizeof(int));
    // cudaMalloc(&dec_maximum_host, num_Elements_host * sizeof(int));

    cudaMemset(dec_vertex_type_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(vertex_type_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(delta_counter_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(all_min_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(all_max_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(all_saddle_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(DS_M_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(AS_M_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(dec_DS_M_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(dec_AS_M_host, -1,  num_Elements_host * sizeof(int));
    
    cudaMemset(wrong_neighbors_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(updated_vertex_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_neighbors_index_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_rank_max_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_rank_max_index_host, -1,  num_Elements_host * 2 * sizeof(int));
    cudaMemset(wrong_rank_saddle_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_rank_saddle_index_host, -1,  num_Elements_host * 2 * sizeof(int));
    cudaMemset(wrong_rank_max_2_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_rank_max_index_2_host, -1,  num_Elements_host * 2 * sizeof(int));
    
    cudaMemset(wrong_neighbors_ds_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_neighbors_ds_index_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_rank_min_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_rank_min_index_host, -1,  num_Elements_host * 2 * sizeof(int));
    cudaMemset(wrong_rank_saddle_join_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_rank_saddle_join_index_host, -1,  num_Elements_host * 2 * sizeof(int));
    cudaMemset(wrong_rank_min_2_host, -1,  num_Elements_host * sizeof(int));
    cudaMemset(wrong_rank_min_index_2_host, -1,  num_Elements_host * 2 * sizeof(int));
    cudaMemset(lowerStars_host, -1,  (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    // cudaMemset(dec_lowerStars_host, -1,  (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    cudaMemset(upperStars_host, -1,  (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    // cudaMemset(dec_upperStars_host, -1,  (maxNeighbors_host+1) * num_Elements_host * sizeof(int));
    
    // cudaMemset(adjacency_host, -1,  maxNeighbors_host * num_Elements_host * sizeof(int));
    cudaMemset(minimum_host, -1, num_Elements_host * sizeof(int));
    // cudaMemset(dec_minimum_host, -1, num_Elements_host * sizeof(int));
    cudaMemset(saddles1_host, -1, num_Elements_host * sizeof(int));
    // cudaMemset(dec_saddles1_host, -1, num_Elements_host * sizeof(int));
    cudaMemset(saddles2_host, -1, num_Elements_host * sizeof(int));
    // cudaMemset(dec_saddles2_host, -1, num_Elements_host * sizeof(int));
    cudaMemset(maximum_host, -1, num_Elements_host * sizeof(int));
    // cudaMemset(dec_maximum_host, -1, num_Elements_host * sizeof(int));

    

    cudaMemcpyToSymbol(dec_vertex_type, &dec_vertex_type_host, sizeof(int*));
    cudaMemcpyToSymbol(updated_vertex, &updated_vertex_host, sizeof(int*));
    cudaMemcpyToSymbol(vertex_type, &vertex_type_host, sizeof(int*));
    cudaMemcpyToSymbol(vertex_cells, &vertex_cells_host, sizeof(int*));
    cudaMemcpyToSymbol(all_min, &all_min_host, sizeof(int*));
    cudaMemcpyToSymbol(all_max, &all_max_host, sizeof(int*));
    cudaMemcpyToSymbol(all_saddle, &all_saddle_host, sizeof(int*));
    cudaMemcpyToSymbol(delta_counter, &delta_counter_host, sizeof(int*));
    cudaMemcpyToSymbol(DS_M, &DS_M_host, sizeof(int*));
    cudaMemcpyToSymbol(AS_M, &AS_M_host, sizeof(int*));
    cudaMemcpyToSymbol(dec_DS_M, &dec_DS_M_host, sizeof(int*));
    cudaMemcpyToSymbol(dec_AS_M, &dec_AS_M_host, sizeof(int*));
    cudaMemcpyToSymbol(lowerStars, &lowerStars_host, sizeof(int*));
    cudaMemcpyToSymbol(upperStars, &upperStars_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_lowerStars, &dec_lowerStars_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_upperStars, &dec_upperStars_host, sizeof(int*));
    // cudaMemcpyToSymbol(adjacency, &adjacency_host, sizeof(int*));
    cudaMemcpyToSymbol(minimum, &minimum_host, sizeof(int*));
    cudaMemcpyToSymbol(saddles1, &saddles1_host, sizeof(int*));
    cudaMemcpyToSymbol(saddles2, &saddles2_host, sizeof(int*));
    cudaMemcpyToSymbol(maximum, &maximum_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_minimum, &dec_minimum_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_saddles1, &dec_saddles1_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_saddles2, &dec_saddles2_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_maximum, &dec_maximum_host, sizeof(int*));


    cudaMemcpyToSymbol(width, &width_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(height, &height_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(depth, &depth_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(maxNeighbors, &maxNeighbors_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num_Elements, &num_Elements_host, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(numFaces, &numFaces_host, sizeof(int), 0, cudaMemcpyHostToDevice);

    
    cudaMemcpyToSymbol(wrong_neighbors, &wrong_neighbors_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_neighbors_index, &wrong_neighbors_index_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_max, &wrong_rank_max_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_max_index, &wrong_rank_max_index_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_max_2, &wrong_rank_max_2_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_max_index_2, &wrong_rank_max_index_2_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_saddle, &wrong_rank_saddle_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_saddle_index, &wrong_rank_saddle_index_host, sizeof(int*));


    
    cudaMemcpyToSymbol(wrong_neighbors_ds, &wrong_neighbors_ds_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_neighbors_ds_index, &wrong_neighbors_ds_index_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_min, &wrong_rank_min_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_min_index, &wrong_rank_min_index_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_min_2, &wrong_rank_min_2_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_min_index_2, &wrong_rank_min_index_2_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_saddle_join, &wrong_rank_saddle_join_host, sizeof(int*));
    cudaMemcpyToSymbol(wrong_rank_saddle_join_index, &wrong_rank_saddle_join_index_host, sizeof(int*));

    cudaMemcpyToSymbol(bound, &host_bound, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(thr, &host_thre, sizeof(float), 0, cudaMemcpyHostToDevice);
    
    std::cout<<host_thre<<std::endl;
    dim3 blockSize(256);
    dim3 gridSize((num_Elements_host + blockSize.x - 1) / blockSize.x);

    // computeAdjacency<<<gridSize, blockSize>>>();
    // cudaDeviceSynchronize();
    // computeAdjacency_host();
    // return 0;
    

    dim3 blockDim1(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 gridDim1((width_host + TILE_SIZE - 1) / TILE_SIZE,
                 (height_host + TILE_SIZE - 1) / TILE_SIZE,
                 (depth_host + TILE_SIZE - 1) / TILE_SIZE);
    generateLookupTable<<<1,1>>>();
    // auto start = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    // 3. 创建 GPU 指针
    // int* d_lookupTable;

    // // 4. 在 GPU 端分配内存
    // cudaMalloc((void**)&d_lookupTable, SIZE);

    // // 5. 复制 `lookupTable` 到 `d_lookupTable`
    // cudaMemcpyFromSymbol(d_lookupTable, lookupTable, SIZE, 0, cudaMemcpyDeviceToDevice);

    // // 6. 从 `GPU` 复制到 `CPU`
    // cudaMemcpy(lookupTable_host, d_lookupTable, SIZE, cudaMemcpyDeviceToHost);

    // // 7. 释放 GPU 内存
    // cudaFree(d_lookupTable);

    // int* d_adjacency;

    // // 4. 在 GPU 端分配内存
    // cudaMalloc((void**)&d_adjacency, 14 * num_Elements_host * sizeof(int));

    // // 5. 复制 `lookupTable` 到 `d_lookupTable`
    // cudaMemcpyFromSymbol(d_adjacency, adjacency, SIZE, 0, cudaMemcpyDeviceToDevice);

    // // 6. 从 `GPU` 复制到 `CPU`
    // cudaMemcpy(adjacency_host, d_adjacency, SIZE, cudaMemcpyDeviceToHost);

    // // 7. 释放 GPU 内存
    // cudaFree(d_adjacency);
    
    // calculateLUT(nullptr, 0, 1<<4, 0, 0);
    // calculateLUT(nullptr, 12, 1<<4, 1<<4, 0);
    // calculateLUT(nullptr, 15, 1<<4, 2 * (1<<4), 0);
    // calculateLUT(nullptr, 48, 1<<4, 3 * (1<<4), 0);
    // calculateLUT(nullptr, 51, 1<<4, 4 * (1<<4), 0);
    // calculateLUT(nullptr, 63, 1<<4, 5 * (1<<4), 0);
    // std::cout<<"end"<<std::endl;
    // calculateLUT(nullptr, 4, 1<<6, 6 * (1<<4), 0);
    // calculateLUT(nullptr, 13, 1<<6, 6 * (1<<4) + (1<<6), 0);
    // calculateLUT(nullptr, 16, 1<<6, 6 * (1<<4) + 2 * (1<<6), 0);
    // calculateLUT(nullptr, 31, 1<<6, 6 * (1<<4) + 3 * (1<<6), 0);
    // calculateLUT(nullptr, 49, 1<<6, 6 * (1<<4) + 4 * (1<<6), 0);
    // calculateLUT(nullptr, 55, 1<<6, 6 * (1<<4) + 5 * (1<<6), 0);
    // std::cout<<"end"<<std::endl;
    
    // calculateLUT(nullptr, 3, 1<<7, 6 * (1<<4) + 6 * (1 << 6), 0);
    // calculateLUT(nullptr, 60, 1<<7, 6 * (1<<4) + 6 * (1 << 6) + (1 << 7), 0);
    // std::cout<<"end"<<std::endl;
    // calculateLUT(nullptr, 1, 1<<8, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7), 0);
    // calculateLUT(nullptr, 7, 1<<8, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + (1<<8), 0);
    // calculateLUT(nullptr, 19, 1<<8, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 2 * (1<<8), 0);
    // calculateLUT(nullptr, 28, 1<<8, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 3 * (1<<8), 0);
    // calculateLUT(nullptr, 52, 1<<8, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 4 * (1<<8), 0);
    // calculateLUT(nullptr, 61, 1<<8, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 5 * (1<<8), 0);
    // std::cout<<"end"<<std::endl;
    // calculateLUT(nullptr, 5, 1<<10, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8), 0);
    // calculateLUT(nullptr, 17, 1<<10, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + (1<<10), 0);
    // calculateLUT(nullptr, 20, 1<<10, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 2 * (1<<10), 0);
    // calculateLUT(nullptr, 23, 1<<10, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 3 * (1<<10), 0);
    // calculateLUT(nullptr, 29, 1<<10, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 4 * (1<<10), 0);
    // calculateLUT(nullptr, 53, 1<<10, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 5 * (1<<10), 0);
    // std::cout<<"end"<<std::endl;
    
    // calculateLUT(nullptr, 21, 1<<14, 6 * (1<<4) + 6 * (1 << 6) + 2* (1 << 7) + 6 * (1<<8) + 6 * (1<<10), 0);
    // std::cout<<"end"<<std::endl;

    // std::ofstream outFile("LUT.bin", std::ios::binary);
    // if (!outFile) {
    //     std::cerr << "无法创建文件 LUT.bin" << std::endl;
    //     return 1;
    // }

    // // 3. 将数组写入文件
    // outFile.write(reinterpret_cast<const char*>(LUT), sizeof(LUT));

    // // 4. 关闭文件
    // outFile.close();
    // // cudaDeviceSynchronize();
    // std::cout<<"wrting completed"<<std::endl;
    loadLUTToGPU();
    cudaDeviceSynchronize();

    // return 0;
    // std::cout << "LUT 已保存为 LUT.bin" << std::endl;
    
    cudaEventRecord(start, 0);
    classifyVertex_CUDA<<<gridDim1, blockDim1>>>(0, 0);
    cudaDeviceSynchronize();
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    std::cout<<"vertex classification: "<<elapsedTime/1000<<std::endl;
    // return 0;
    init_Manifold<<<gridSize, blockSize>>>(0);
    cudaDeviceSynchronize();
    init_Manifold<<<gridSize, blockSize>>>(1);
    cudaDeviceSynchronize();

    ComputeDirection<<<gridSize, blockSize>>>(0);
    cudaDeviceSynchronize();
    
    ComputeDescendingManifold<<<gridSize, blockSize>>>(0);
    cudaDeviceSynchronize();
    
    ComputeAscendingManifold<<<gridSize, blockSize>>>(0);
    cudaDeviceSynchronize();
    
    ExtractCP<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    
    int nSaddle2_host = 0;
    int nMin_host = 0;
    int nMax_host = 0;
    int nSaddle1_host = 0;

    cudaMemcpyFromSymbol(&nSaddle2_host, nSaddle2, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nSaddle1_host, nSaddle1, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nMax_host, nMax, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nMin_host, nMin, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout<<nSaddle2_host<< ", "<<nSaddle1_host<<", "<<nMax_host<<", "<<nMin_host<<std::endl;
    // return 0;

    dim3 gridSize_2saddle((nSaddle2_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_max((nMax_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_min((nMin_host + blockSize.x - 1) / blockSize.x);
    dim3 gridSize_1saddle((nSaddle1_host + blockSize.x - 1) / blockSize.x);
    
    
    // initialize saddletriples;
    int *saddleTriplets_host, *dec_saddleTriplets_host, *saddle1Triplets_host, *dec_saddle1Triplets_host, *tempArray_host, *dec_tempArray_host, *tempArrayMin_host, *dec_tempArrayMin_host;
    int *largestSaddlesForMax_host, *dec_largestSaddlesForMax_host, *smallestSaddlesForMin_host, *dec_smallestSaddlesForMin_host;
    int *dec_saddlerank_host, *saddlerank_host, *dec_saddle1rank_host, *saddle1rank_host;
    cudaMalloc(&saddleTriplets_host, nSaddle2_host * 46 * sizeof(int));
    cudaMalloc(&dec_saddleTriplets_host, nSaddle2_host * 46 * sizeof(int));
    cudaMalloc(&saddle1Triplets_host, nSaddle1_host * 46 * sizeof(int));
    cudaMalloc(&dec_saddle1Triplets_host, nSaddle1_host * 46 * sizeof(int));

    cudaMalloc(&dec_saddlerank_host, nSaddle2_host * sizeof(int));
    cudaMalloc(&saddlerank_host, nSaddle2_host * sizeof(int));
    cudaMalloc(&dec_saddle1rank_host, nSaddle1_host * sizeof(int));
    cudaMalloc(&saddle1rank_host, nSaddle1_host * sizeof(int));
    
    cudaMalloc(&reachable_saddle_for_Max_host, 45 * nMax_host * sizeof(int));
    cudaMalloc(&dec_reachable_saddle_for_Max_host, 45 * nMax_host * sizeof(int));

    cudaMalloc(&reachable_saddle_for_Min_host, 45 * nMin_host * sizeof(int));
    cudaMalloc(&dec_reachable_saddle_for_Min_host, 45 * nMin_host * sizeof(int));
    cudaMalloc(&max_index_host, num_Elements_host * sizeof(int));
    
    // cudaMalloc(&tempArray_host, num_Elements_host * sizeof(int));
    // cudaMalloc(&dec_tempArray_host, num_Elements_host * sizeof(int));
    // cudaMalloc(&tempArrayMin_host, num_Elements_host * sizeof(int));
    // cudaMalloc(&dec_tempArrayMin_host, num_Elements_host * sizeof(int));
    // cudaMalloc(&largestSaddlesForMax_host, nMax_host * sizeof(int));
    // cudaMalloc(&dec_largestSaddlesForMax_host, nMax_host * sizeof(int));
    // cudaMalloc(&smallestSaddlesForMin_host, nMin_host * sizeof(int));
    // cudaMalloc(&dec_smallestSaddlesForMin_host, nMin_host * sizeof(int));

    // cudaMemset(dec_saddlerank_host, -1, nSaddle2_host * sizeof(int));
    // cudaMemset(saddlerank_host, -1, nSaddle2_host *  sizeof(int));
    // cudaMemset(dec_saddle1rank_host, -1, nSaddle1_host * sizeof(int));
    // cudaMemset(saddle1rank_host, -1, nSaddle1_host *  sizeof(int));

    cudaMemset(saddle1Triplets_host, -1, nSaddle1_host * 46 * sizeof(int));
    
    // cudaMemset(tempArray_host, -1, num_Elements_host * sizeof(int));
    // cudaMemset(dec_tempArray_host, -1, num_Elements_host * sizeof(int));
    // cudaMemset(tempArrayMin_host, -1, num_Elements_host * sizeof(int));
    // cudaMemset(dec_tempArrayMin_host, -1, num_Elements_host * sizeof(int));
    
    // cudaMemset(largestSaddlesForMax_host,-1, nMax_host * sizeof(int));
    // cudaMemset(dec_largestSaddlesForMax_host, -1,nMax_host * sizeof(int));
    // cudaMemset(smallestSaddlesForMin_host,-1, nMin_host * sizeof(int));
    // cudaMemset(dec_smallestSaddlesForMin_host, -1, nMin_host * sizeof(int));

    cudaMemset(reachable_saddle_for_Max_host, 0, 45 * nMax_host * sizeof(int));
    cudaMemset(dec_reachable_saddle_for_Max_host, 0, 45 * nMax_host * sizeof(int));

    cudaMemset(reachable_saddle_for_Min_host, 0, 45 * nMin_host * sizeof(int));
    cudaMemset(dec_reachable_saddle_for_Min_host, 0, 45 * nMin_host * sizeof(int));

    checkCudaError(cudaMemset(max_index_host, 0, num_Elements_host * sizeof(int)), "error");

    cudaMemcpyToSymbol(saddleTriplets, &saddleTriplets_host, sizeof(int*));
    cudaMemcpyToSymbol(dec_saddleTriplets, &dec_saddleTriplets_host, sizeof(int*));
    cudaMemcpyToSymbol(saddle1Triplets, &saddle1Triplets_host, sizeof(int*));
    cudaMemcpyToSymbol(dec_saddle1Triplets, &dec_saddle1Triplets_host, sizeof(int*));
   
    cudaMemcpyToSymbol(reachable_saddle_for_Max, &reachable_saddle_for_Max_host, sizeof(int*));
    cudaMemcpyToSymbol(dec_reachable_saddle_for_Max, &dec_reachable_saddle_for_Max_host, sizeof(int*));

    cudaMemcpyToSymbol(reachable_saddle_for_Min, &reachable_saddle_for_Min_host, sizeof(int*));
    cudaMemcpyToSymbol(dec_reachable_saddle_for_Min, &dec_reachable_saddle_for_Min_host, sizeof(int*));

    cudaMemcpyToSymbol(max_index, &max_index_host, sizeof(int*));
    // cudaMemset(reachable_saddle_for_Max_host, -1, nSaddle2_host * nMax_host * sizeof(int));
    // cudaMemcpyToSymbol(tempArray, &tempArray_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_tempArray, &dec_tempArray_host, sizeof(int*));
    // cudaMemcpyToSymbol(tempArrayMin, &tempArrayMin_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_tempArrayMin, &dec_tempArrayMin_host, sizeof(int*));

    // cudaMemcpyToSymbol(largestSaddlesForMax, &largestSaddlesForMax_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_largestSaddlesForMax, &dec_largestSaddlesForMax_host, sizeof(int*));
    // cudaMemcpyToSymbol(smallestSaddlesForMin, &smallestSaddlesForMin_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_smallestSaddlesForMin, &dec_smallestSaddlesForMin_host, sizeof(int*));

    // cudaMemcpyToSymbol(dec_saddlerank, &dec_saddlerank_host, sizeof(int*));
    // cudaMemcpyToSymbol(saddlerank, &saddlerank_host, sizeof(int*));
    // cudaMemcpyToSymbol(dec_saddle1rank, &dec_saddle1rank_host, sizeof(int*));
    // cudaMemcpyToSymbol(saddle1rank, &saddle1rank_host, sizeof(int*));
    

    

    // sortCP(minimum_host, maximum_host, saddles1_host, saddles2_host, 
    //         saddlerank_host, saddle1rank_host,
    //         input_data_host, nMin_host, nMax_host, 
    //         nSaddle1_host, nSaddle2_host);
    // sortCP(dec_minimum_host, dec_maximum_host, dec_saddles1_host, dec_saddles2_host, 
    //         decp_data_host, nMin_host, nMax_host, 
    //         nSaddle1_host, nSaddle2_host);
    // ComputeTempArray<<<gridSize_max, blockSize>>>(1, 1);
    // ComputeTempArray<<<gridSize_min, blockSize>>>(0, 1);
    // computeMaxIndex<<<gridSize_max, blockSize>>>();
    // cudaDeviceSynchronize();
    // computeMinIndex<<<gridSize_min, blockSize>>>();
    // cudaDeviceSynchronize();
    
    findAscPaths<<<gridSize_2saddle, blockSize>>>(1);
    cudaDeviceSynchronize();
    findDescPaths<<<gridSize_1saddle, blockSize>>>(1);
    cudaDeviceSynchronize();
    
    // sortTripletsOnGPU(saddleTriplets_host, nSaddle2_host, input_data_host);
    // sortTripletsOnGPU(saddle1Triplets_host, nSaddle1_host, input_data_host);

    // computelargestSaddlesForMax<<<gridSize_max, blockSize>>>();
    // cudaDeviceSynchronize();
    // computesmallestSaddlesForMin<<<gridSize_min, blockSize>>>();
    // cudaDeviceSynchronize();
    // return 0;
    // compute_Max_for_Saddle<<<gridSize_2saddle, blockSize>>>();
    // compute_Min_for_Saddle<<<gridSize_1saddle, blockSize>>>();
    
    cudaDeviceSynchronize();
    
    
    int host_count_f_max = 0;
    int host_count_f_min = 0;
    int host_count_f_saddle = 0;
    // dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    // dim3 gridDim((width_host + TILE_SIZE - 1) / TILE_SIZE,
    //              (height_host + TILE_SIZE - 1) / TILE_SIZE,
    //              (depth_host + TILE_SIZE - 1) / TILE_SIZE);
    c_loops(gridSize, blockSize, host_count_f_max, host_count_f_min, host_count_f_saddle);
    cudaDeviceSynchronize();
    std::cout<<"cloops ended!"<<std::endl;
    
    init_Manifold<<<gridSize, blockSize>>>(1);
    cudaDeviceSynchronize();
    ComputeDirection<<<gridSize, blockSize>>>(1);
    cudaDeviceSynchronize();
    
    
    ComputeDescendingManifold<<<gridSize, blockSize>>>(1);
    ComputeAscendingManifold<<<gridSize, blockSize>>>(1);

    cudaMemcpyToSymbol(dec_nSaddle1, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(dec_nSaddle2, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(dec_nMax, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(dec_nMin, &initialValue, sizeof(int));
    // ExtractCP<<<gridSize, blockSize>>>(1);
    // cudaDeviceSynchronize();

    // sortCP(dec_minimum_host, dec_maximum_host, dec_saddles1_host, dec_saddles2_host, 
    //         dec_saddlerank_host, dec_saddle1rank_host,
    //         decp_data_host, nMin_host, nMax_host, 
    //         nSaddle1_host, nSaddle2_host);

    // ComputeTempArray<<<gridSize_max, blockSize>>>(1);
    // ComputeTempArray<<<gridSize_min, blockSize>>>(0);
    init_max_saddle<<<gridSize_max, blockSize>>>();
    cudaDeviceSynchronize();
    init_min_saddle<<<gridSize_min, blockSize>>>();
    cudaDeviceSynchronize();
    findAscPaths<<<gridSize_2saddle, blockSize>>>();
    cudaDeviceSynchronize();
    findDescPaths<<<gridSize_1saddle, blockSize>>>();
    cudaDeviceSynchronize();

    // sortTripletsOnGPU(dec_saddleTriplets_host, nSaddle2_host, decp_data_host);
    // sortTripletsOnGPU(dec_saddle1Triplets_host, nSaddle1_host, decp_data_host);

    // computelargestSaddlesForMax<<<gridSize_max, blockSize>>>(1);
    // computesmallestSaddlesForMin<<<gridSize_min, blockSize>>>(1);
    
    init_neighbor_buffer<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    get_wrong_split_neighbors<<<gridSize_2saddle, blockSize>>>();
    get_wrong_join_neighbors<<<gridSize_1saddle, blockSize>>>();
    cudaDeviceSynchronize();

    // return 0;
    int host_number_of_false_cases, host_number_of_false_cases1;
    cudaMemcpyFromSymbol(&host_number_of_false_cases, num_false_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_number_of_false_cases1, num_false_cases1, sizeof(int), 0, cudaMemcpyDeviceToHost);

    int host_wrong_max_counter, host_wrong_max_counter_2;

    std::cout<<host_number_of_false_cases<<std::endl;
    
    if(host_number_of_false_cases == 0){
        
        s_loops(gridSize, blockSize, gridSize_2saddle);
        cudaMemcpyFromSymbol(&host_wrong_max_counter, wrong_max_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
        if(host_wrong_max_counter == 0){
            saddle_loops(gridSize_max, gridSize, blockSize);
        }  
    }
    
    if(host_number_of_false_cases1 == 0 && preserve_join_tree){
        
        s_loops_join(gridSize, blockSize, gridSize_1saddle);
        
        int host_wrong_min_counter = 0;
        cudaMemcpyFromSymbol(&host_wrong_min_counter, wrong_min_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
        if(host_wrong_min_counter == 0){
            saddle_loops_join(gridSize_min, gridSize, blockSize);
        }
        
    }


    int host_wrong_min_counter = 0, host_wrong_min_counter_2 = 0, host_wrong_saddle_counter = 0, host_wrong_saddle_counter_join = 0;
    cudaMemcpyFromSymbol(&host_number_of_false_cases, num_false_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_number_of_false_cases1, num_false_cases1, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_saddle, count_f_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_wrong_min_counter, wrong_min_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_wrong_max_counter, wrong_max_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_wrong_min_counter_2, wrong_min_counter_2, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_wrong_max_counter_2, wrong_max_counter_2, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_wrong_saddle_counter, wrong_saddle_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_wrong_saddle_counter_join, wrong_saddle_counter_join, sizeof(int), 0, cudaMemcpyDeviceToHost);
    // return 0;
    std::vector<std::vector<float>> time_counter;
    while(host_number_of_false_cases > 0 || host_number_of_false_cases1 > 0|| host_count_f_max > 0 || host_count_f_min > 0 || host_count_f_saddle > 0 
            || host_wrong_max_counter > 0 || host_wrong_max_counter_2 > 0 || host_wrong_saddle_counter > 0
            || host_wrong_min_counter > 0 || host_wrong_min_counter_2 > 0 || host_wrong_saddle_counter_join > 0){
        std::vector<float> temp_time;
        std::cout<<
        "whole loops:"<<host_number_of_false_cases<<", "<<host_number_of_false_cases1<<", "<<
        host_count_f_max <<", "<< host_count_f_min << ", "<< host_count_f_saddle<<", "
        <<host_wrong_max_counter<<", "<<host_wrong_max_counter_2<<", "<<host_wrong_saddle_counter
        <<", "<<host_wrong_min_counter<<", "<<host_wrong_min_counter_2<<", "<<host_wrong_saddle_counter_join<<std::endl;
        
        float cloops_sub = 0.0;
        float rloops_sub = 0.0;
        float manifold_sub = 0.0;
        float saddle_t_sub = 0.0;
        float saddle_m_sub = 0.0;
        float extract_wrong_saddle_sub = 0.0;
        float s_loops_sub = 0.0;
        float saddle_loops_sub = 0.0;
        

        cudaEventRecord(start, 0);
        c_loops(gridSize, blockSize, host_count_f_max, host_count_f_min, host_count_f_saddle);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cloops_sub+=elapsedTime/1000;
        
        cudaEventRecord(start, 0);
        r_loops(gridSize, blockSize,host_number_of_false_cases, host_number_of_false_cases1);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        rloops_sub+=elapsedTime/1000;


        cudaEventRecord(start, 0);
        init_Manifold<<<gridSize, blockSize>>>(1);
        cudaDeviceSynchronize();
        ComputeDirection<<<gridSize, blockSize>>>(1);
        cudaDeviceSynchronize();

        ComputeDescendingManifold<<<gridSize, blockSize>>>(1);
        ComputeAscendingManifold<<<gridSize, blockSize>>>(1);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        manifold_sub+=elapsedTime/1000;
        // ExtractCP<<<gridSize, blockSize>>>(1);
        // cudaDeviceSynchronize();

        // sortCP(dec_minimum_host, dec_maximum_host, dec_saddles1_host, dec_saddles2_host, 
        //         dec_saddlerank_host, dec_saddle1rank_host,
        //         decp_data_host, nMin_host, nMax_host, 
        //         nSaddle1_host, nSaddle2_host);

        // ComputeTempArray<<<gridSize_max, blockSize>>>(1);
        // ComputeTempArray<<<gridSize_min, blockSize>>>(0);
        cudaEventRecord(start, 0);
        init_max_saddle<<<gridSize_max, blockSize>>>();
        cudaDeviceSynchronize();
        init_min_saddle<<<gridSize_min, blockSize>>>();
        cudaDeviceSynchronize();
        findAscPaths<<<gridSize_2saddle, blockSize>>>();
        findDescPaths<<<gridSize_1saddle, blockSize>>>();
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        saddle_t_sub+=elapsedTime/1000;
        // sortTripletsOnGPU(dec_saddleTriplets_host, nSaddle2_host, decp_data_host);
        // sortTripletsOnGPU(dec_saddle1Triplets_host, nSaddle1_host, decp_data_host);

        // cudaEventRecord(start, 0);
        // computelargestSaddlesForMax<<<gridSize_max, blockSize>>>(1);
        // computesmallestSaddlesForMin<<<gridSize_min, blockSize>>>(1);
        // cudaDeviceSynchronize();
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        saddle_m_sub+=elapsedTime/1000;
        // compute the saddle labels for the decp_data;
        cudaMemcpyToSymbol(num_false_cases, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(num_false_cases1, &initialValue, sizeof(int));
        init_saddle_buffer<<<gridSize, blockSize>>>(0);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start, 0);
        get_wrong_split_neighbors<<<gridSize_2saddle, blockSize>>>();
        get_wrong_join_neighbors<<<gridSize_1saddle, blockSize>>>();
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        extract_wrong_saddle_sub+=elapsedTime/1000;
        
        cudaMemcpyFromSymbol(&host_number_of_false_cases, num_false_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_number_of_false_cases1, num_false_cases1, sizeof(int), 0, cudaMemcpyDeviceToHost);
        // std::cout<<"host_wrong neighbors: "<<host_number_of_false_cases<<std::endl;
        if(host_number_of_false_cases == 0){
            // std::cout<<"host_wrong_max_counter: "<<host_wrong_max_counter<<std::endl;
            cudaEventRecord(start, 0);
            s_loops(gridSize, blockSize, gridSize_2saddle);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            s_loops_sub+=elapsedTime/1000;
            if(host_wrong_max_counter == 0){
                cudaEventRecord(start, 0);
                saddle_loops(gridSize_max, gridSize, blockSize);
                cudaDeviceSynchronize();
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                saddle_loops_sub+=elapsedTime/1000;
                cudaDeviceSynchronize();
            }  
        }
        
        if(host_number_of_false_cases1 == 0 && preserve_join_tree){
            cudaEventRecord(start, 0);
            s_loops_join(gridSize, blockSize, gridSize_1saddle);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            s_loops_sub+=elapsedTime/1000;

            int host_wrong_min_counter = 0;
            cudaMemcpyFromSymbol(&host_wrong_min_counter, wrong_min_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            if(host_wrong_min_counter == 0){
                cudaEventRecord(start, 0);
                saddle_loops_join(gridSize_min, gridSize, blockSize);
                cudaDeviceSynchronize();
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                saddle_loops_sub+=elapsedTime/1000;
                
            }
            
        }
        
        cudaEventRecord(start, 0);
        c_loops(gridSize, blockSize, host_count_f_max, host_count_f_min, host_count_f_saddle);;

        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cloops_sub+=elapsedTime/1000;
       
        cudaEventRecord(start, 0);
        init_Manifold<<<gridSize, blockSize>>>(1);
        cudaDeviceSynchronize();
        ComputeDirection<<<gridSize, blockSize>>>(1);
        cudaDeviceSynchronize();
        

        ComputeDescendingManifold<<<gridSize, blockSize>>>(1);
        ComputeAscendingManifold<<<gridSize, blockSize>>>(1);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        manifold_sub+=elapsedTime/1000;
        // ExtractCP<<<gridSize, blockSize>>>(1);
        // cudaDeviceSynchronize();

        // sortCP(dec_minimum_host, dec_maximum_host, dec_saddles1_host, dec_saddles2_host, 
        //         dec_saddlerank_host, dec_saddle1rank_host,
        //         decp_data_host, nMin_host, nMax_host, 
        //         nSaddle1_host, nSaddle2_host);

        // ComputeTempArray<<<gridSize_max, blockSize>>>(1);
        // ComputeTempArray<<<gridSize_min, blockSize>>>(0);

        cudaEventRecord(start, 0);
        init_max_saddle<<<gridSize_max, blockSize>>>();
        cudaDeviceSynchronize();
        init_min_saddle<<<gridSize_min, blockSize>>>();
        cudaDeviceSynchronize();
        findAscPaths<<<gridSize_2saddle, blockSize>>>();
        findDescPaths<<<gridSize_1saddle, blockSize>>>();
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        saddle_t_sub+=elapsedTime/1000;
        // sortTripletsOnGPU(dec_saddleTriplets_host, nSaddle2_host, decp_data_host);
        // sortTripletsOnGPU(dec_saddle1Triplets_host, nSaddle1_host, decp_data_host);
        cudaEventRecord(start, 0);
        // computelargestSaddlesForMax<<<gridSize_max, blockSize>>>(1);
        // computesmallestSaddlesForMin<<<gridSize_min, blockSize>>>(1);
        
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        saddle_m_sub+=elapsedTime/1000;
        // compute the saddle labels for the decp_data;

        cudaMemcpyToSymbol(num_false_cases, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(num_false_cases1, &initialValue, sizeof(int));
        init_saddle_buffer<<<gridSize, blockSize>>>(0);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start, 0);
        get_wrong_split_neighbors<<<gridSize_2saddle, blockSize>>>();
        get_wrong_join_neighbors<<<gridSize_1saddle, blockSize>>>();
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        extract_wrong_saddle_sub+=elapsedTime/1000;

        cudaMemcpyFromSymbol(&host_number_of_false_cases, num_false_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_number_of_false_cases1, num_false_cases1, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
        if(host_number_of_false_cases == 0){
            cudaEventRecord(start, 0);
            s_loops(gridSize, blockSize, gridSize_2saddle);
            
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            s_loops_sub+=elapsedTime/1000;
            cudaMemcpyFromSymbol(&host_wrong_max_counter, wrong_max_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            if(host_wrong_max_counter == 0){
                cudaEventRecord(start, 0);
                saddle_loops(gridSize_max, gridSize, blockSize);
                cudaDeviceSynchronize();
                
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                saddle_loops_sub+=elapsedTime/1000;
            }  
        }
        
        if(host_number_of_false_cases1 == 0 && preserve_join_tree){
            cudaEventRecord(start, 0);
            s_loops_join(gridSize, blockSize, gridSize_1saddle);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            s_loops_sub+=elapsedTime/1000;
            int host_wrong_min_counter = 0;
            cudaMemcpyFromSymbol(&host_wrong_min_counter, wrong_min_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
            if(host_wrong_min_counter == 0){
                cudaEventRecord(start, 0);
                saddle_loops_join(gridSize_min, gridSize, blockSize);
                cudaDeviceSynchronize();
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                saddle_loops_sub+=elapsedTime/1000;
            }
            
        }

        cudaEventRecord(start, 0);
        c_loops(gridSize, blockSize, host_count_f_max, host_count_f_min, host_count_f_saddle);;
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cloops_sub+=elapsedTime/1000;

        cudaEventRecord(start, 0);
        init_Manifold<<<gridSize, blockSize>>>(1);
        cudaDeviceSynchronize();
        ComputeDirection<<<gridSize, blockSize>>>(1);
        cudaDeviceSynchronize();
        
        ComputeDescendingManifold<<<gridSize, blockSize>>>(1);
        ComputeAscendingManifold<<<gridSize, blockSize>>>(1);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        manifold_sub+=elapsedTime/1000;

        cudaEventRecord(start, 0);
        init_max_saddle<<<gridSize_max, blockSize>>>();
        cudaDeviceSynchronize();
        init_min_saddle<<<gridSize_min, blockSize>>>();
        cudaDeviceSynchronize();
        findAscPaths<<<gridSize_2saddle, blockSize>>>();
        findDescPaths<<<gridSize_1saddle, blockSize>>>();
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        saddle_t_sub+=elapsedTime/1000;
        
        cudaEventRecord(start, 0);
        // computelargestSaddlesForMax<<<gridSize_max, blockSize>>>(1);
        // computesmallestSaddlesForMin<<<gridSize_min, blockSize>>>(1);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        saddle_m_sub+=elapsedTime/1000;
        

        cudaMemcpyToSymbol(num_false_cases, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(num_false_cases1, &initialValue, sizeof(int));
        cudaEventRecord(start, 0);
        init_saddle_buffer<<<gridSize, blockSize>>>(0);
        cudaDeviceSynchronize();
        get_wrong_split_neighbors<<<gridSize_2saddle, blockSize>>>();
        get_wrong_join_neighbors<<<gridSize_1saddle, blockSize>>>();
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        extract_wrong_saddle_sub+=elapsedTime/1000;

        cudaMemcpyFromSymbol(&host_number_of_false_cases, num_false_cases, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_number_of_false_cases1, num_false_cases1, sizeof(int), 0, cudaMemcpyDeviceToHost);
        if(host_number_of_false_cases == 0){
            
            cudaMemcpyToSymbol(wrong_max_counter, &initialValue, sizeof(int));
            cudaMemcpyToSymbol(wrong_max_counter_2, &initialValue, sizeof(int));
            init_buffer<<<gridSize, blockSize>>>();
            get_wrong_index_max<<<gridSize_2saddle, blockSize>>>();
            cudaDeviceSynchronize();
            cudaMemcpyToSymbol(wrong_saddle_counter, &initialValue, sizeof(int));
            init_saddle_rank_buffer<<<gridSize, blockSize>>>();
            // get_wrong_index_saddles<<<gridSize_max, blockSize>>>();
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(&host_wrong_saddle_counter, wrong_saddle_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_wrong_max_counter, wrong_max_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_wrong_max_counter_2, wrong_max_counter_2, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
        }

        if(host_number_of_false_cases1 == 0 && preserve_join_tree){
            
            cudaMemcpyToSymbol(wrong_min_counter, &initialValue, sizeof(int));
            cudaMemcpyToSymbol(wrong_min_counter_2, &initialValue, sizeof(int));
            init_buffer<<<gridSize, blockSize>>>(1);
            get_wrong_index_min<<<gridSize_1saddle, blockSize>>>();
            cudaDeviceSynchronize();
            init_saddle_rank_buffer<<<gridSize, blockSize>>>(1);
            cudaMemcpyToSymbol(wrong_saddle_counter_join, &initialValue, sizeof(int));
            // get_wrong_index_saddles_join<<<gridSize_min, blockSize>>>();
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(&host_wrong_saddle_counter_join, wrong_saddle_counter_join, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_wrong_min_counter_2, wrong_min_counter_2, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_wrong_min_counter, wrong_min_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
        }

        temp_time.push_back(cloops_sub);
        temp_time.push_back(rloops_sub);
        temp_time.push_back(manifold_sub);
        temp_time.push_back(saddle_t_sub);
        temp_time.push_back(saddle_m_sub);
        temp_time.push_back(extract_wrong_saddle_sub);
        temp_time.push_back(s_loops_sub);
        temp_time.push_back(saddle_loops_sub);
        
        time_counter.push_back(temp_time);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, startt, stop);
    std::cout<<"whole time:" <<elapsedTime/1000<<std::endl;

    std::ofstream outFilep("../stat_result/performance1_cuda_"+std::to_string(er)+"_"+compressor_id+".txt", std::ios::app);
        // 检查文件是否成功打开
        if (!outFilep) {
            std::cerr << "Unable to open file for writing." << std::endl;
            return 1; // 返回错误码
        }
        
        outFilep << std::setprecision(17)<< "related_error: "<<er << std::endl;
        
        // outFilep << "edit_ratio: "<< ratio << std::endl;  
        int c1 = 0;
        for (const auto& row : time_counter) {
            outFilep << "iteration: "<<c1<<": ";
            for (size_t i = 0; i < row.size(); ++i) {
                outFilep << row[i];
                if (i != row.size() - 1) { // 不在行的末尾时添加逗号
                    outFilep << ", ";
                }
            }
            // 每写完一行后换行
            outFilep << std::endl;
            c1+=1;
        }
        outFilep << "\n"<< std::endl;

    // get_cp_number<<<1,1>>>();
    cudaDeviceSynchronize();

    
    // cudaMemcpy(decp_data_host, decp_data, num_Elements_host * sizeof(float), cudaMemcpyDeviceToHost);
    // 1. 先获取 `decp_data` 设备指针的值
    // float *d_ptr;
    // cudaMemcpyFromSymbol(&d_ptr, decp_data, sizeof(float*), 0, cudaMemcpyDeviceToHost);

    // // 2. 再把 `d_ptr` 指向的设备数据复制到 host
    // cudaMemcpy(decp_data_host, d_ptr, num_Elements_host * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // cudaMemcpy(decp_data_host, d_ptr, num_Elements_host * sizeof(float), cudaMemcpyDeviceToHost);

    // cudaMemcpyFromSymbol(decp_data_host, decp_data, num_Elements_host * sizeof(float), 0, cudaMemcpyDeviceToHost);
    std::vector<float> decp_d;
    decp_d.resize(num_Elements_host);
    cudaMemcpy(decp_d.data(), decp_data_host, num_Elements_host * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<int> delta_counter_d;
    delta_counter_d.resize(num_Elements_host);
    cudaMemcpy(delta_counter_d.data(), delta_counter_host, num_Elements_host * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<float> input_data_d;
    input_data_d.resize(num_Elements_host);
    cudaMemcpy(input_data_d.data(), input_data_host, num_Elements_host * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<float> decp_data_copy_d;
    decp_data_copy_d.resize(num_Elements_host);
    cudaMemcpy(decp_data_copy_d.data(), decp_data_copy, num_Elements_host * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cost("QMC", decp_d.data(), decp_data_copy_d.data(), input_data_d.data(),compressor_id,delta_counter_d);
    
    // std::string filename, float* decp_data, float* decp_data_copy, float* input_data, std::string compressor_id, std::vector<int> delta_counter
    
    saveArrayToBin(decp_data_copy_d.data(), num_Elements_host, "decp.bin");
    saveArrayToBin(decp_d.data(), num_Elements_host, "fixed.bin");

    


}