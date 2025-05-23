#include <iostream>
#include <fstream>
#include <vector>
#include "../include/api/ExtreM.h"
#include "./UnionFind.h"
#include "SZ3/api/sz.hpp"
#include <parallel/algorithm>  
#include <omp.h>
#include <set>
#include <cfloat>
#include <filesystem>
#include <bitset>


extern "C" {
    int directions[42] = {
       1,0,0, -1,0,0, 0,1,0, 0,-1,0,
        0,0,1, 0,0,-1, -1,1,0, 1,-1,0,
        0,1,1, 0,-1,-1, -1,0,1, 1,0,-1,
        -1,1,1, 1,-1,-1
    };
    int neighborOffsets[14][3] = {
        {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0},
        {0,0,1}, {0,0,-1}, {-1,1,0}, {1,-1,0},
        {0,1,1}, {0,-1,-1}, {-1,0,1}, {1,0,-1},
        {-1,1,1}, {1,-1,-1}
    };
    int LUT[1 << 14];
    
    double* d_deltaBuffer;
    int width, height, depth, maxNeighbors, num_Elements;
    int* adjacency;
    double *decp_data, *input_data, *decp_data_copy;
    double bound, additional_time, compression_time, er, maxValue, minValue;
    size_t cmpSize = 0;
    std::string file_path;
    int *dec_vertex_type, *vertex_type;

    std::atomic_int count_f_max;
    std::atomic_int count_f_min;
    std::atomic_int count_f_saddle;
    int number_of_false_cases, wrong_max_counter, wrong_saddle_counter, wrong_saddle_counter_join, wrong_max_counter_2, globalMin, dec_globalMin;
    int number_of_false_cases1, wrong_min_counter, wrong_min_counter_2;
    int *all_max, *all_min, *all_saddle;
    
    std::vector<std::vector<int>> vertex_cells;
    int *or_saddle_max_map, *wrong_neighbors, *wrong_neighbors1, *wrong_neighbors_index, *wrong_rank_max, *wrong_rank_max_index, *wrong_rank_saddle, *wrong_rank_saddle_index, *wrong_rank_max_2, *wrong_rank_max_index_2;
    int *or_saddle_min_map, *wrong_neighbors_ds, *wrong_neighbors_ds_index, *wrong_rank_min, *wrong_rank_min_index, *wrong_rank_min_2, *wrong_rank_min_index_2;
    int *wrong_rank_saddle_join, *wrong_rank_saddle_join_index;

    std::vector<std::array<int, 46>> saddleTriplets, dec_saddleTriplets;
    std::vector<std::array<int, 46>> saddle1Triplets, dec_saddle1Triplets;

    std::vector<int> saddles2, maximum, dec_saddles2, dec_maximum, delta_counter;
    std::vector<int> saddles1, minimum, dec_saddles1, dec_minimum;
    std::unordered_map<int, int> largestSaddlesForMax, dec_largestSaddlesForMax;
    std::unordered_map<int, int> smallestSaddlesForMin, dec_smallestSaddlesForMin;
    
    int *lowerStars, *upperStars, *dec_lowerStars, *dec_upperStars;
    int direction_to_index_mapping[26][3] = 
    {
        {1, 0, 0}, {-1, 0, 0},   
        {0, 1, 0}, {0, -1, 0},
        {-1, 1, 0}, {1, -1, 0}, 
        {0, 0, -1}, {0, -1, 1},
        {0, 0, 1}, {0, 1, -1},
        {-1, 0, 1}, {1, 0, -1},
        {1, 1, 0}, {-1, -1, 0},
        {1, 0, 1}, {-1, 0, -1},
        {0, 1, 1}, {0, -1, -1},
        {1, 1, 1}, {1, 1, -1},  
        {1, -1, 1}, {1, -1, -1},
        {-1, 1, 1}, {-1, 1, -1},
        {-1, -1, 1}, {-1, -1, -1}
    };

    void Branch::print() const {
        std::cout << "Branch:" << std::endl;

        // 打印顶点信息
        std::cout << "  Vertices:" << std::endl;
        for (const auto &vertex : vertices) {
            std::cout << "    Order: " << vertex.first << ", GlobalId: " << vertex.second << std::endl;
        }

        // 打印父分支信息
        if (parentBranch) {
            std::cout << "  Parent Branch's first vertex GlobalId: "
                    << parentBranch->vertices[0].second << std::endl;
        } else {
            std::cout << "  No Parent Branch" << std::endl;
        }
    }

    void getdata(const std::string &filename, 
                const double er, 
                int data_size, std::string type = "split") {
        
        
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (size != static_cast<std::streamsize>(data_size * sizeof(double))) {
            std::cerr << "File size does not match expected data size." << std::endl;
            return;
        }

        file.read(reinterpret_cast<char *>(input_data), size);
        if (!file) {
            std::cerr << "Error reading file." << std::endl;
            return;
        }

        
       
        SZ3::Config conf(data_size); 
        conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
        conf.errorBoundMode = SZ3::EB_REL;
        conf.relErrorBound = er; 

        
        
        char *compressedData = SZ_compress(conf, input_data, cmpSize);

        SZ_decompress(conf, compressedData, cmpSize, decp_data);
        std::copy(decp_data, decp_data + data_size, decp_data_copy);

        delete[] compressedData;

        
        minValue = *std::min_element(input_data, input_data + data_size);
        maxValue = *std::max_element(input_data, input_data + data_size);
        bound = (maxValue - minValue) * er;
        
        std::cout << "Data read, compressed, and decompressed successfully." << std::endl;
    }


    int ComputeDescendingManifold(const double *offset, const int *vertex_type, int *DS_M){
        
        #pragma omp parallel for
        for (int i = 0; i < num_Elements; i++) {
            int largest_neighbor = i;
            for(int j = 0; j< maxNeighbors; j++)
            {
                int neighbor_id = adjacency[ maxNeighbors * i + j];
                if(neighbor_id == -1) continue;
                if(offset[largest_neighbor] < offset[neighbor_id] || (offset[largest_neighbor] == offset[neighbor_id] && largest_neighbor < neighbor_id)) largest_neighbor = neighbor_id;
            }
            // if(i == 60111) std::cout<<"id is:"<<largest_neighbor<<std::endl;
            DS_M[i] = largest_neighbor;
        }

        
        for (int i = 0; i < num_Elements; ++i) {
            int v = i;
            bool is_saddle_neighbor = false;
            for(int j = 0; j<maxNeighbors; j++){
                int neighborId = adjacency[i*maxNeighbors+j];
                if(neighborId == -1) continue;
                if(vertex_type[neighborId] == 2 || vertex_type[3]){
                    is_saddle_neighbor = true;
                    break;
                }
            }

            if(!is_saddle_neighbor) continue;
            while(true){
                int u = DS_M[v];
                int w = DS_M[u];
                if (u == w) break;
                DS_M[v] = w;
            }
            
        }

        return 0;
    };

    void getlabel(int* DS_M, int* AS_M, const int* direction_as, const int* direction_ds, int& un_sign_ds, int& un_sign_as){
    
        un_sign_ds = 0;
        un_sign_as = 0;

        #pragma omp parallel for reduction(+:un_sign_as) reduction(+:un_sign_ds)
        
        for(int i=0;i<num_Elements;i++){
            int cur = AS_M[i];
            if (cur!=-1 and direction_as[cur]!=-1){
                
                int direc = direction_as[cur];
                AS_M[i] = direc;
                if (direction_as[AS_M[i]] != -1){
                    un_sign_as+=1;  
                }
                
            }

        
            cur = DS_M[i];
            
            if (cur!=-1 and DS_M[i]!=-1){
                
                int direc = direction_ds[cur];
                DS_M[i] = direc;
                if (direction_ds[DS_M[i]]!=-1){
                    un_sign_ds+=1;
                    }
                } 
        }
        return;

    }

    std::vector<uint8_t> encodeTo3BitBitmap(const std::vector<int>& data) {
        std::vector<uint8_t> bitmap; // 存储结果的位图
        uint8_t currentByte = 0; // 当前字节
        int bitIndex = 0; // 当前位的索引

        for (int value : data) {
            if (value < 0 || value > 6) {
                
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

    double calculateMSE(const double* original, const double* compressed) {


        double mse = 0.0;
        for (size_t i = 0; i < num_Elements; i++) {
            mse += std::pow(static_cast<double>(original[i]) - compressed[i], 2);
        }
        mse /= num_Elements;
        return mse;
    }

    double calculatePSNR( double* original, double* compressed, double maxValue) {
        double mse = calculateMSE(original, compressed);
        if (mse == 0) {
            return std::numeric_limits<double>::infinity(); // Perfect match
        }
        double psnr = -20.0*log10(sqrt(mse)/maxValue);
        return psnr;
    }

    void cost(std::string filename, double* decp_data, double* decp_data_copy, double* input_data, std::string compressor_id, std::vector<int> delta_counter){
        std::vector<int> indexs(num_Elements, 0);
        std::vector<double> edits;
        std::vector<unsigned long long> exponents;
        std::vector<unsigned long long> mantissas;

        int cnt = 0;
        for (int i=0;i<num_Elements;i++){
            if (decp_data_copy[i]!=decp_data[i]){
                indexs[i] = delta_counter[i];
                
                if(indexs[i]==6){
                    edits.push_back(-(decp_data_copy[i] - (input_data[i] - bound)));
                }
                cnt++;
            }
        }
        
        
        std::vector<uint8_t> bitmap = encodeTo3BitBitmap(indexs);

        
        std::string indexfilename = "/pscratch/sd/y/yuxiaoli/MSCz/data1"+filename+std::to_string(bound)+".bin";
        std::string editsfilename = "/pscratch/sd/y/yuxiaoli/MSCz/data_edits"+filename+std::to_string(bound)+".bin";
        std::string compressedindex = "/pscratch/sd/y/yuxiaoli/MSCz/data1"+filename+std::to_string(bound)+".bin.zst";
        std::string compressededits = "/pscratch/sd/y/yuxiaoli/MSCz/data_edits"+filename+std::to_string(bound)+".bin.zst";
        
        // Shockwave, 64, 64, 512

        double ratio = double(cnt)/(num_Elements);
        std::cout<<cnt<<","<<ratio<<std::endl;

        std::ofstream file(indexfilename, std::ios::binary | std::ios::out);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(bitmap.data()), bitmap.size());
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
            file1.write(reinterpret_cast<const char*>(edits.data()), edits.size()*sizeof(double));
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

        double overall_ratio = double(original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
    
        double bitRate = 64/overall_ratio; 

        double psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
        double fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);

        std::ofstream outFile3("./stat_result/result_"+filename+"_"+compressor_id+"_detailed_additional_time.txt", std::ios::app);

        
        if (!outFile3) {
            std::cerr << "Unable to open file for writing." << std::endl;
            return; // 返回错误码
        }

        
        outFile3 << std::to_string(bound)<<":" << std::endl;
        outFile3 << std::setprecision(17)<< "related_error: "<<er << std::endl;
        outFile3 << std::setprecision(17)<< "OCR: "<<overall_ratio << std::endl;
        outFile3 <<std::setprecision(17)<< "CR: "<<double(original_dataSize)/compressed_dataSize << std::endl;
        outFile3 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
        outFile3 << std::setprecision(17)<<"BR: "<< 64/(double(original_indexSize)/compressed_dataSize) << std::endl;
        outFile3 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
        outFile3 << std::setprecision(17)<<"fixed_psnr: "<<fixed_psnr << std::endl;
        

        
        outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
        outFile3 << std::setprecision(17)<<"compression_time: "<<compression_time<< std::endl;
        outFile3 << std::setprecision(17)<<"additional_time: "<<additional_time<< std::endl;
        outFile3 << "\n" << std::endl;

        outFile3.close();

        std::cout << "Variables have been appended to output.txt" << std::endl;
        return;
    };

    std::string extractFilename(const std::string& path) {
    
        size_t lastSlash = path.find_last_of("/\\");
        std::string filename = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);

        size_t dotPos = filename.find_last_of('.');
        std::string name = (dotPos == std::string::npos) ? filename : filename.substr(0, dotPos);

        return name;
    };

    void mappath(int* DS_M, int* AS_M, const double* offset){
        int h_un_sign_as = num_Elements;
        int h_un_sign_ds = num_Elements;
        int *direction_as, *direction_ds;
        direction_as = new int[num_Elements];
        direction_ds = new int[num_Elements];
        #pragma omp parallel for
        for (int i = 0; i < num_Elements; i++) {
            int largest_neighbor = i;
            int smallest_neighbor = i;
            for(int j = 0; j< maxNeighbors; j++)
            {
                int neighbor_id = adjacency[ maxNeighbors * i + j];
                if(neighbor_id == -1) continue;
                if(offset[largest_neighbor] > offset[neighbor_id] || (offset[largest_neighbor] == offset[neighbor_id] && largest_neighbor > neighbor_id)) largest_neighbor = neighbor_id;

                
                if(offset[smallest_neighbor] < offset[neighbor_id] || (offset[smallest_neighbor] == offset[neighbor_id] && smallest_neighbor < neighbor_id)) smallest_neighbor = neighbor_id;
            }

            direction_ds[i] = largest_neighbor == i? -1 : largest_neighbor;
            direction_as[i] = smallest_neighbor == i?-1 : smallest_neighbor;
        }
        
        
        while(h_un_sign_as>0 or h_un_sign_ds>0){
            
            h_un_sign_as=0;
            h_un_sign_ds=0;
            
            getlabel(DS_M, AS_M, direction_as, direction_ds, h_un_sign_as, h_un_sign_ds);
            
        }   

        return;
    };

    int computePathCompressionSingle(
        int *const segmentation,
        const bool computeAscending,
        const double *const offset) {
        
        std::vector<int> activeVertices;

        for (int i = 0; i < num_Elements; i++) {
            bool hasLargerNeighbor = false;
            int &mi = segmentation[i];
            mi = i;
            for(int j = 0; j< maxNeighbors; j++)
            {
                int neighbor_id = adjacency[ maxNeighbors * i + j];
                if(neighbor_id == -1) continue;

                if (computeAscending) {
                    
                    if(offset[mi] > offset[neighbor_id] || (offset[mi] == offset[neighbor_id] && mi > neighbor_id)) {
                        mi = neighbor_id;
                        hasLargerNeighbor = true;
                    }
                } else {
                    if(offset[mi] < offset[neighbor_id] || (offset[mi] == offset[neighbor_id] && mi < neighbor_id))
                    {
                        mi = neighbor_id;
                        hasLargerNeighbor = true;
                    }
                }
                // }

                if (hasLargerNeighbor) {
                    activeVertices.push_back(i);
                }
            }
        }

        size_t nActiveVertices = activeVertices.size();
        size_t currentIndex = 0;

        // 压缩路径直到没有变化
        while (nActiveVertices > 0) {
            for (size_t i = 0; i < nActiveVertices; i++) {
                int const &v = activeVertices[i];
                int &vMan = segmentation[v];

                // 压缩路径
                vMan = segmentation[vMan];

                // 检查是否已完全压缩
                if (vMan != segmentation[vMan]) {
                    activeVertices[currentIndex++] = v;
                }
            }

            nActiveVertices = currentIndex;
            currentIndex = 0;
        }

        return 0; // 返回成功
    }


    int ComputeAscendingManifold(const double *offset, const int *vertex_type, int *AS_M){
       
        #pragma omp parallel for
        for (int i = 0; i < num_Elements; i++) {
            int largest_neighbor = i;
            for(int j = 0; j< maxNeighbors; j++)
            {
                int neighbor_id = adjacency[ maxNeighbors * i + j];
                if(neighbor_id == -1) continue;
                if(offset[largest_neighbor] > offset[neighbor_id] || (offset[largest_neighbor] == offset[neighbor_id] && largest_neighbor > neighbor_id)) largest_neighbor = neighbor_id;
            }

            AS_M[i] = largest_neighbor;
        }

        
        for (int i = 0; i < num_Elements; ++i) {
            int v = i;
            bool is_saddle_neighbor = false;
            for(int j = 0; j<maxNeighbors; j++){
                int neighborId = adjacency[i*maxNeighbors+j];
                if(neighborId == -1) continue;
                if(vertex_type[neighborId] == 1 || vertex_type[3]){
                    is_saddle_neighbor = true;
                    break;
                }
            }

            if(!is_saddle_neighbor) continue;
            while(true){
                int u = AS_M[v];
                int w = AS_M[u];
                if (u == w) break;
                AS_M[v] = w;
            }
            
        }

        return 0;
    };

    void computeAdjacency() {
        int data_size = width * height * depth;
        #pragma omp parallel for

        for(int i=0;i<data_size;i++){
            int y = (i / (width)) % height; // Get the x coordinate
            int x = i % width; // Get the y coordinate
            int z = (i / (width * height)) % depth;
            int neighborIdx = 0;
            for (int d = 0; d < maxNeighbors; d++) {
                
                int dirX = directions[d * 3];     
                int dirY = directions[d * 3 + 1]; 
                int dirZ = directions[d * 3 + 2]; 
                int newX = x + dirX;
                int newY = y + dirY;
                int newZ = z + dirZ;
                int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
                
                if (newX >= 0 && newX < width && newY >= 0 && newY < height && r < width*height*depth && newZ<depth && newZ>=0) {
                    
                    adjacency[i * maxNeighbors + neighborIdx] = r;
                    neighborIdx++;

                }
            // Fill the remaining slots with -1 or another placeholder value
            for (int j = neighborIdx; j < maxNeighbors; ++j) {
                adjacency[i * maxNeighbors + j] = -1;
                }
            }
        
        }
    }
    
    bool isless(const int v, const int u, const double* offset){
        return offset[v] < offset[u] || (std::abs(offset[v] - offset[u]) == 0 && v < u);
    }

    bool islarger(const int v, const int u, const double *offset){
        return offset[v] > offset[u] || (std::abs(offset[v] - offset[u]) == 0 && v > u);
    }

    void sortAndRemoveDuplicates(std::array<int, 46> &triplet) {
      std::sort(triplet.begin(), triplet.begin() + triplet[44]);
      int tempPointer = 1;
      for(int p = 1; p < triplet[44]; p++) {
        if(triplet[p - 1] != triplet[p]) { // if we have a new value, we step ahead
            triplet[tempPointer] = triplet[p];
            tempPointer++;
        }
      }
      triplet[44] = tempPointer;
    }

    void getVerticesFromTriangleID1(
        int triangleID,
        int &v1, 
        int &v2, 
        int &v3) {


        int baseID, min_x, min_y, min_z;
        int starID = triangleID;
        
        
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
            // if(starID == 2945) printf("type1: %d %d %d\n", v1, v2, v3);
            return;
            
        }


        // if(starID == 2945) printf("type4: %d %d %d %d %d %d\n", v1, v2, v3, starID, triangleID, subTriangleType);
        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        
        return;
        
    }
        bool isAdjacent(const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
        int index = p1.first+p1.second*width;
        int index2 = p2.first+p2.second*width;

        bool isadjacent=false;
        for(int i =0;i<maxNeighbors;i++){
            int j = adjacency[index*maxNeighbors+i];
            if(j==-1) continue;
            if(j==index2){
                isadjacent=true;
                return isadjacent;
            }
        }

        for(int i =0;i<maxNeighbors;i++){
            int j = adjacency[index2*maxNeighbors+i];
            if(j==-1) continue;
            if(j==index){
                isadjacent=true;
                return isadjacent;
            }
        }
        return isadjacent;
    }

    std::vector<std::vector<int> > findWedges(const std::vector<int>& star) {
        std::vector<std::vector<int> > wedges;
        std::set<int> visited;

        for (const auto& vertex : star) {
            if (visited.count(vertex) == 0) {
                std::vector<int> wedge;
                std::queue<int> q;
                q.push(vertex);
                visited.insert(vertex);

                while (!q.empty()) {
                    auto current = q.front();
                    q.pop();
                    wedge.push_back(current);

                    for (const auto& neighbor : star) 
                    {
                        if (isAdjacent(std::make_pair(current%width, current/width), std::make_pair(neighbor%width, neighbor/width)) && visited.count(neighbor) == 0) 
                        {
                            q.push(neighbor);
                            visited.insert(neighbor);
                        }
                    }
                }
                wedges.push_back(wedge);
            }
        }

        return wedges;
    }

// 辅助函数：找到并查集的根



    int classifyVertex1(const double* heightMap, int i, int *lowerStars, int *upperStars, int type=0) {
    
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
        double currentHeight = heightMap[i];
        std::vector<int > lowerStar, upperStar;
        int vertexId = i;
        int y = (i / (width)) % height; // Get the x coordinate
        int x = i % width; // Get the y coordinate
        int z = (i / (width * height)) % depth;
        
        lowerStars[vertexId * (maxNeighbors+1) + maxNeighbors] = 0;
        upperStars[vertexId * (maxNeighbors+1) + maxNeighbors] = 0;
        for (int d =0;d<maxNeighbors;d++) {
            if(adjacency[vertexId*maxNeighbors+d]==-1) continue;
            int r = adjacency[i*maxNeighbors+d];
            
            if (heightMap[r] < currentHeight) {
                lowerStars[vertexId * (maxNeighbors+1) + lowerStars[vertexId * (maxNeighbors+1) + maxNeighbors]] = r;
                lowerStars[vertexId * (maxNeighbors+1) + maxNeighbors]++;
                lowerStar.emplace_back(r);
            } 
            else if (heightMap[r] == currentHeight and r<i) {
                lowerStars[vertexId * (maxNeighbors+1) + lowerStars[vertexId * (maxNeighbors+1) + maxNeighbors]] = r;
                lowerStars[vertexId * (maxNeighbors+1) + maxNeighbors]++;
                lowerStar.emplace_back(r);
            }

            if (heightMap[r] > currentHeight) {
                upperStars[vertexId * (maxNeighbors+1) + upperStars[vertexId * (maxNeighbors+1) + maxNeighbors]] = r;
                upperStars[vertexId * (maxNeighbors+1) + maxNeighbors]++;
                upperStar.emplace_back(r);
            } 
            else if (heightMap[r] == currentHeight and r>i) {
                upperStars[vertexId * (maxNeighbors+1) + upperStars[vertexId * (maxNeighbors+1) + maxNeighbors]] = r;
                upperStars[vertexId * (maxNeighbors+1) + maxNeighbors]++;
                upperStar.emplace_back(r);
            }
        }
        
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
        
        std::vector<int> tr;
        tr = vertex_cells[vertexId];


        int const vertexStarSize = tr.size();
        
        for(int i = 0; i < vertexStarSize; i++) {
            int cellId = tr[i];
            int const cellSize = 3;
            int v1, v2, v3;
            getVerticesFromTriangleID1(cellId,  v1, v2, v3);
            std::vector<int> vertices{v1, v2, v3};
            for(int j = 0; j < cellSize; j++) {

            int neighborId0 = vertices[j];
            if(neighborId0 == vertexId) continue;
                bool const lower0 = heightMap[neighborId0] < currentHeight || (heightMap[neighborId0] == currentHeight and neighborId0<vertexId);
                // connect it to everybody except himself and vertexId
                for(int k = j + 1; k < cellSize; k++) {
                    int neighborId1 = vertices[k];
                    if((neighborId1 != neighborId0) && (neighborId1 != vertexId)) {
                        
                        bool const lower1 = heightMap[neighborId1] < currentHeight || (heightMap[neighborId1] == currentHeight and neighborId1<vertexId);

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
                                if((*neighbors)[l] == neighborId0) {
                                    
                                    lowerId0 = l;
                                }
                                if((*neighbors)[l] == neighborId1) {
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
            }
        };

        
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
        if(lowerComponentNumber == 0 && upperComponentNumber == 1) return 0;
        else if(lowerComponentNumber == 1 && upperComponentNumber == 0) return 4;
        else if(lowerComponentNumber == 1 && upperComponentNumber == 1) return 5;
        else if(lowerComponentNumber > 1 && upperComponentNumber > 1) return 3;
        else if(lowerComponentNumber > 1) return 1;
        else if(upperComponentNumber > 1) return 2;
        
            

        return 5;
    }

    int calculateNeighborIndex(int nx, int ny, int nz){
        for(int i = 0; i<maxNeighbors; i++){
            if(nx == neighborOffsets[i][0] && ny == neighborOffsets[i][1] && nz == neighborOffsets[i][2]) return i;
        }
        return -1;
    }

    

    int calculateLUT(const double* heightMap, int i, int type=0) {
        const int tableSize = 1 << 14;
        for(int config = 0; config<tableSize;config++){

            std::bitset<14> binary(config); 
            if(config<10) std::cout<<binary<<std::endl;
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
            int vertexId = i;
            int y = (i / (width)) % height; // Get the x coordinate
            int x = i % width; // Get the y coordinate
            int z = (i / (width * height)) % depth;
            
        
            for (int d =0;d<maxNeighbors;d++) {
                if(adjacency[vertexId*maxNeighbors+d]==-1) continue;
                int r = adjacency[i*maxNeighbors+d];
                
                if (binary[d]==0) {
                    lowerStar.emplace_back(r);
                } 
                else{
                    upperStar.emplace_back(r);
                }
            }
            
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
            
            std::vector<int> tr;
            tr = vertex_cells[vertexId];


            int const vertexStarSize = tr.size();
            
            for(int i = 0; i < vertexStarSize; i++) {
                int cellId = tr[i];
                int const cellSize = 3;
                int v1, v2, v3;
                getVerticesFromTriangleID1(cellId,  v1, v2, v3);
                std::vector<int> vertices{v1, v2, v3};
                for(int j = 0; j < cellSize; j++) {

                int neighborId0 = vertices[j];
                
                if(neighborId0 == vertexId) continue;
                int nx = neighborId0 % width;
                int ny = (neighborId0 / width) % height;
                int nz = (neighborId0 / (width * height)) % depth;
                int idx = calculateNeighborIndex(nx - x, ny - y, nz -z);
                if(idx == -1) std::cout<<"wrong1" <<std::endl;
                    bool const lower0 = binary[idx] == 0;
                    // connect it to everybody except himself and vertexId
                    for(int k = j + 1; k < cellSize; k++) {
                        int neighborId1 = vertices[k];
                        if((neighborId1 != neighborId0) && (neighborId1 != vertexId)) {
                            int nx = neighborId1 % width;
                            int ny = (neighborId1 / width) % height;
                            int nz = (neighborId1 / (width * height)) % depth;
                            int idx = calculateNeighborIndex(nx - x, ny - y, nz - z);
                            if(idx == -1) std::cout<<"wrong" <<std::endl;
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
                                    if((*neighbors)[l] == neighborId0) {
                                        
                                        lowerId0 = l;
                                    }
                                    if((*neighbors)[l] == neighborId1) {
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
                }
            };

            
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
            if(lowerComponentNumber == 0 && upperComponentNumber == 1) LUT[config]=0;
            else if(lowerComponentNumber == 1 && upperComponentNumber == 0) LUT[config]=4;
            else if(lowerComponentNumber == 1 && upperComponentNumber == 1) LUT[config]=5;
            else if(lowerComponentNumber > 1 && upperComponentNumber > 1) LUT[config]=3;
            else if(lowerComponentNumber > 1) LUT[config]=1;
            else if(upperComponentNumber > 1) LUT[config]=2;
        }      
    }
    

    void classifyAllVertices(int *vertex_type_tmp, const double* heightMap, 
                             int *lowerStars, int *upperStars,
                             int type=0) {

        
        #pragma omp parallel for
        for (int i = 0; i < num_Elements; ++i) {
            vertex_type_tmp[i] = classifyVertex1(heightMap, i, lowerStars, upperStars, type);
        }
    }

    void init_delta(){
        #pragma omp parallel for
        for(int tid = 0; tid < num_Elements; tid++){
            d_deltaBuffer[tid] = -4.0 * bound;
            // delta_counter[tid] = 0;
        }    
    }

    bool areVectorsDifferent(const std::vector<int>& vec1, const std::vector<int>& vec2) {
        if (vec1.size() != vec2.size()) {
            return true; 
        }

        for (size_t i = 0; i < vec1.size(); ++i) {
            if (vec1[i] != vec2[i]) {
                return true; 
            }
        }

        return false; 
    }
    double atomicCASDouble(double* address, double expected, double desired) {
        // reinterpret_cast 将 double 转换为 uint64_t 以便原子操作
        uint64_t* address_as_ull = reinterpret_cast<uint64_t*>(address);
        uint64_t expected_ull = *reinterpret_cast<uint64_t*>(&expected);
        uint64_t desired_ull = *reinterpret_cast<uint64_t*>(&desired);

        // 使用 std::atomic_compare_exchange 强制类型转换
        std::atomic<uint64_t>* atomic_address = reinterpret_cast<std::atomic<uint64_t>*>(address_as_ull);
        atomic_address->compare_exchange_strong(expected_ull, desired_ull);

        // 返回操作前的值
        return *reinterpret_cast<double*>(&expected_ull);
    }

    void swap(int index, double delta) {
        int update_successful = 0;

        while (update_successful == 0) {
            double current_value = d_deltaBuffer[index];

            if (std::abs(delta) > current_value) {
                
                double swapped = atomicCASDouble(&d_deltaBuffer[index], current_value, delta);
                if (swapped == current_value) {
                    update_successful = 1; 
                }
            } else {
                update_successful = 1; 
            }
        }
    }

    void c_loop(int index, int direction = 0, std::string type = "split"){      
        // preservation of split tree?->decrease f
        if(type == "split"){
            if (direction == 0){
                
                // if vertex is a regular point.
                if (vertex_type[index]!=4){
                    
                    double d = ((input_data[index] - bound) + decp_data[index]) / 2.0 - decp_data[index];
                    double oldValue = d_deltaBuffer[index];
                    
                    if (d > oldValue) {
                        swap(index, d);
                    }  

                    return;
                
                }
                else{
                    
                    // if is a maximum in the original data;
                    int largest_index = index;
                    
                    for(int i = 0; i< maxNeighbors;i++)
                    {
                        int neighbor = adjacency[maxNeighbors * index + i];
                        if(neighbor == -1) continue;
                        if(islarger(neighbor, largest_index, decp_data))
                        {
                            largest_index = neighbor;
                        }
                    }
                    
                    if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
                        return;
                    }

                    double d = ((input_data[largest_index] - bound) + decp_data[largest_index]) / 2.0 - decp_data[largest_index];
                    
                    double oldValue = d_deltaBuffer[largest_index];
                    if (d > oldValue) {
                        swap(largest_index, d);
                    }  

                    return;
                }
            }
            
            else if (direction != 0){
                
                if (vertex_type[index]!=0){
                    int smallest_index = index;
                    for(int i = 0; i<maxNeighbors;i++)
                    {
                        int neighbor = adjacency[maxNeighbors * index + i];
                        if(neighbor == -1) continue;
                        if(isless(neighbor, smallest_index, input_data))
                        {
                            smallest_index = neighbor;
                        }
                    }
                    double d = ((input_data[smallest_index] - bound) + decp_data[smallest_index]) / 2.0 - decp_data[smallest_index];
                    if(decp_data[index]>decp_data[smallest_index] or (decp_data[index]==decp_data[smallest_index] and index>smallest_index)){
                        return;
                    }
                    double oldValue = d_deltaBuffer[smallest_index];
                    if (d > oldValue) {
                        swap(smallest_index, d);
                    }  
                    return;
                
                }
            
                else{
                    double d = ((input_data[index] - bound) + decp_data[index]) / 2.0 - decp_data[index];
                    double oldValue = d_deltaBuffer[index];
                    if (d > oldValue) {
                        swap(index, d);
                    } 
                    return;
                }

                
            }    
        }
        // preservation of join tree?->increase -f
        else{
            if (direction == 0){
                
                // if vertex is a regular point, need to increase its largest neighbor.
                if (vertex_type[index]!=4){
                    int largest_index = index;
                    
                    for(int i = 0; i< maxNeighbors; i++){
                        int neighbor = adjacency[maxNeighbors * index + i];
                        if(neighbor == -1) continue;
                        if(islarger(neighbor, largest_index, input_data)){
                            largest_index = neighbor;
                        }
                    }
                    
                    if(decp_data[index]<decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index<largest_index)){
                        return;
                    }

                    double d = ((input_data[largest_index] + bound) + decp_data[largest_index]) / 2.0 - decp_data[largest_index];
                    
                    double oldValue = d_deltaBuffer[largest_index];
                    if (d > oldValue) {
                        swap(largest_index, d);
                    }  

                    return;
                
                }
                else{
                    
                    // if is a maximum in the original data, need to increase itself;
                    double d = ((input_data[index] + bound) + decp_data[index]) / 2.0 - decp_data[index];
                    double oldValue = d_deltaBuffer[index];
                    
                    if (d > oldValue) {
                        swap(index, d);
                    }  

                    return;
                    
                }
            }
            
            else if (direction != 0){
                // if a regular point in the original data, need to increase itself.
                if (vertex_type[index]!=0){
                    
                    double d = ((input_data[index] + bound) + decp_data[index]) / 2.0 - decp_data[index];
                    double oldValue = d_deltaBuffer[index];
                    if (d > oldValue) {
                        swap(index, d);
                    } 
                    return;
                }
            
                else{
                    
                    int smallest_index = index;
                    for(int i = 0; i<maxNeighbors;i++){
                        int neighbor = adjacency[maxNeighbors * index + i];
                        if(neighbor == -1) continue;
                        if(isless(neighbor, smallest_index, decp_data)){
                            smallest_index = neighbor;
                        }
                    }
                    double d = ((input_data[smallest_index] + bound) + decp_data[smallest_index]) / 2.0 - decp_data[smallest_index];
                    
                    if(decp_data[index]<decp_data[smallest_index] or (decp_data[index]==decp_data[smallest_index] and index<smallest_index)){
                        return;
                    }
                    
                    double oldValue = d_deltaBuffer[smallest_index];
                    if (d > oldValue) {
                        swap(smallest_index, d);
                    }  
                    return;
                }

                
            }  
        }

        return;
    }

    void get_false_criticle_points(){
    

        count_f_max=0;
        count_f_min=0;
        count_f_saddle=0;

        #pragma omp parallel for
        for (auto i = 0; i < num_Elements; i ++) {
            int type1 = dec_vertex_type[i];
            
            if(type1 != vertex_type[i]){
                
                // maximum
                if((type1==4 and vertex_type[i]!=4) or (type1!=4 and vertex_type[i]==4)){
                    int idx_fp_max = std::atomic_fetch_add(&count_f_max, 1);
                    all_max[idx_fp_max] = i;
                    
                } 
                if((type1==0 and vertex_type[i]!=0) or (type1!=0 and vertex_type[i]==0)){
                    int idx_fp_min = std::atomic_fetch_add(&count_f_min, 1);
                    all_min[idx_fp_min] = i;
                }
                if((type1==2 and vertex_type[i]!=2) or (type1!=2 and vertex_type[i]==2)){
                    int idx_fp_saddle = std::atomic_fetch_add(&count_f_saddle, 1);
                    all_saddle[idx_fp_saddle] = i;
                }

                if((type1==1 and vertex_type[i]!=1) or (type1!=1 and vertex_type[i]==1)){
                    int idx_fp_saddle = std::atomic_fetch_add(&count_f_saddle, 1);
                    all_saddle[idx_fp_saddle] = i;
                }

                if((type1==3 and vertex_type[i]!=3) or (type1!=3 and vertex_type[i]==3)){
                    int idx_fp_saddle = std::atomic_fetch_add(&count_f_saddle, 1);
                    all_saddle[idx_fp_saddle] = i;
                }


            }

            else if(type1==2 and vertex_type[i]==2 || type1== 1 and vertex_type[i]==1 || type1== 3 and vertex_type[i]==3){
                    // std::vector<int> lowerStar, upperStar, or_lowerStar, or_upperStar;
                    double currentHeight = decp_data[i];
                    double or_currentHeight = input_data[i];
                    std::vector<int> lowerStar(dec_lowerStars + (maxNeighbors+1) * i, dec_lowerStars + (maxNeighbors+1) * i + dec_lowerStars[i * (maxNeighbors + 1) + maxNeighbors]);
                    std::vector<int> upperStar(dec_upperStars + (maxNeighbors+1) * i, dec_upperStars + (maxNeighbors+1) * i + dec_upperStars[i * (maxNeighbors + 1) + maxNeighbors]);
                    std::vector<int> or_lowerStar(lowerStars + (maxNeighbors+1) * i, lowerStars + (maxNeighbors+1) * i + lowerStars[i * (maxNeighbors + 1) + maxNeighbors]);
                    std::vector<int> or_upperStar(upperStars + (maxNeighbors+1) * i, upperStars + (maxNeighbors+1) * i + upperStars[i * (maxNeighbors + 1) + maxNeighbors]);
                    
                    if(areVectorsDifferent(lowerStar, or_lowerStar) or areVectorsDifferent(upperStar, or_upperStar)){
                        int idx_fp_saddle = std::atomic_fetch_add(&count_f_saddle, 1);
                        all_saddle[idx_fp_saddle] = i;
                    }
            }
            
            
        }

    }

    std::vector<int> findDifferences(const std::vector<int>& vec1, const std::vector<int>& vec2) {
        std::set<int> set1(vec1.begin(), vec1.end());
        std::set<int> set2(vec2.begin(), vec2.end());
        std::vector<int> differences;

        for (const auto& elem : set2) {
            if (set1.find(elem) == set1.end()) {
                differences.push_back(elem);
            }
        }

        return differences;
    }

    std::vector<int> findDifference(const std::vector<int>& a, const std::vector<int>& b) {
        // 将两个向量转换为 std::set
        std::set<int> setA(a.begin(), a.end());
        std::set<int> setB(b.begin(), b.end());

        std::vector<int> difference;

        // 使用 std::set_difference 找到 a 中有但 b 中没有的元素
        std::set_difference(setA.begin(), setA.end(), setB.begin(), setB.end(), std::back_inserter(difference));

        return difference;
    }

    void get_false_stars(int index, std::string type="split"){
        // std::vector<int > lowerStar, upperStar;
        double delta;
        std::vector<int> lowerStar(dec_lowerStars + (maxNeighbors+1) * index, dec_lowerStars + (maxNeighbors+1) * index + dec_lowerStars[index * (maxNeighbors + 1) + maxNeighbors]);
        std::vector<int> upperStar(dec_upperStars + (maxNeighbors+1) * index, dec_upperStars + (maxNeighbors+1) * index + dec_upperStars[index * (maxNeighbors + 1) + maxNeighbors]);
        std::vector<int> o_lowerStar(lowerStars + (maxNeighbors+1) * index, lowerStars + (maxNeighbors+1) * index + lowerStars[index * (maxNeighbors + 1) + maxNeighbors]);
        std::vector<int> o_upperStar(upperStars + (maxNeighbors+1) * index, upperStars + (maxNeighbors+1) * index + upperStars[index * (maxNeighbors + 1) + maxNeighbors]);
        
        if(type == "split"){
            if(areVectorsDifferent(lowerStar, o_lowerStar)){
                std::vector<int> diff = findDifferences(lowerStar,o_lowerStar);
                
                // decrease value of the negative lower star.
                    
                for(int i:diff){
                    delta = (input_data[i]-bound) - decp_data[i];
                    double oldValue = d_deltaBuffer[i];
                    if (delta > oldValue) {
                        swap(i, delta);
                    }  

                }

                diff = findDifferences(o_lowerStar, lowerStar);
                if(diff.size() > 0){
                    delta = (input_data[index]-bound) - decp_data[index];
                    double oldValue = d_deltaBuffer[index];
                    if (delta > oldValue) {
                        swap(index, delta);
                    } 
                }
            
            }
        }
        else{
            if(areVectorsDifferent(upperStar, o_upperStar)){
                
                std::vector<int> diff = findDifferences(upperStar, o_upperStar);
                // decrease value of the negative upper star.
                for(int i:diff){
                    delta = (input_data[i]+bound) - decp_data[i];
                    if(index==1239) std::cout<<i<<":"<<delta<<", "<<input_data[i]+bound<<", "<<decp_data[i]+delta<<std::endl;
                    double oldValue = d_deltaBuffer[i];
                    if (delta > oldValue) {
                        swap(i, delta);
                    } 
                }
                // if(diff.size() > 0){
                //     delta = (input_data[index]+bound) - decp_data[index];
                //     double oldValue = d_deltaBuffer[index];
                //     if (delta > oldValue) {
                //         swap(index, delta);
                //     }  
                // }
                
                diff = findDifferences(o_upperStar, upperStar);
                if(diff.size() > 0){
                    delta = (input_data[index]+bound) - decp_data[index];
                    double oldValue = d_deltaBuffer[index];
                    if (delta > oldValue) {
                        swap(index, delta);
                    }  
                }
                
            
            }
        }
        

        return;
        
        
    }

    void apply_delta(std::string type){
        #pragma omp parallel for
        for(int tid = 0; tid < num_Elements; tid++){
            if(d_deltaBuffer[tid] != -4.0 * bound){
                
                if(std::abs(d_deltaBuffer[tid]) > 1e-15 && delta_counter[tid]<5 && std::abs(input_data[tid] - decp_data[tid] - d_deltaBuffer[tid])<=bound) {
                    decp_data[tid] += d_deltaBuffer[tid];
                    delta_counter[tid] += 1;
                }
                else{
                    if(type == "split") decp_data[tid] = input_data[tid] - bound;
                    else decp_data[tid] = input_data[tid] + bound;
                    delta_counter[tid] = 6;
                }
                
            }
        }
    }

    void c_loops(std::string type = "split"){
        float vertexClassification = 0.0;
        float get_fcp = 0.0;
        float c_loopt = 0.0;
        float init_deltat = 0.0;
        float apply_deltat = 0.0;

        auto start = std::chrono::high_resolution_clock::now();
        classifyAllVertices(dec_vertex_type, decp_data, dec_lowerStars, dec_upperStars, 1);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        vertexClassification += duration.count();

        start = std::chrono::high_resolution_clock::now();
        get_false_criticle_points();    
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        get_fcp += duration.count();
        
        while(count_f_max>0 or count_f_min>0 or count_f_saddle>0){
            // std::cout<<"c_loops: "<<count_f_max<<", "<<count_f_min<<", "<<count_f_saddle<<std::endl;
            
            start = std::chrono::high_resolution_clock::now();
            init_delta();
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            init_deltat += duration.count();


            start = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for(auto i = 0; i < count_f_max; i ++){
                
                int critical_i = all_max[i];
                
                c_loop(critical_i, 0, type);

            }                
                
            #pragma omp parallel for
            for(auto i = 0; i < count_f_min; i ++){

                int critical_i = all_min[i];
                c_loop(critical_i, 1, type);
            }
                
            #pragma omp parallel for
            for(int i =0;i<count_f_saddle;i++){
                int index = all_saddle[i];
                get_false_stars(index, type);
            }
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            c_loopt += duration.count();

            start = std::chrono::high_resolution_clock::now();
            apply_delta(type);
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            apply_deltat += duration.count();

            start = std::chrono::high_resolution_clock::now();
            classifyAllVertices(dec_vertex_type, decp_data, dec_lowerStars, dec_upperStars, 1);
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            vertexClassification += duration.count();


            start = std::chrono::high_resolution_clock::now();
            get_false_criticle_points();    
            
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            get_fcp += duration.count();

        }
        float w_time = vertexClassification + get_fcp + c_loopt + init_deltat + apply_deltat;
        
    }

    void get_vertex_traingle(){
        int numFaces = 2 * (width - 1) * (height - 1);
        for(int i = 0; i<numFaces; i++){
            int v1, v2, v3;
            
            getVerticesFromTriangleID1(i, v1, v2, v3);
            vertex_cells[v1].push_back(i);
            vertex_cells[v2].push_back(i);
            vertex_cells[v3].push_back(i);
        }
    }

    void findAscPaths(  std::vector<std::array<int, 46>> &saddleTriplets,
                        std::vector<int> &maximaLocalToGlobal,
                        std::vector<int> &saddlesLocalToGlobal,
                        const int *saddles2, const int *maximum,
                        const double *offset,
                        const int *desManifold, const int *ascManifold, int &counter_tmp){
    
        for(int i = 0; i < saddlesLocalToGlobal.size(); i++) {
            const int vertexId = saddles2[i];
            
            auto &triplet = saddleTriplets[i];
            
            triplet[44] = 0;
            
            for(int k = 0; k < maxNeighbors; k++)
            {
                const int neighborId = adjacency[maxNeighbors * vertexId + k];
                
                if(neighborId == -1) continue;
                if(islarger(neighborId, vertexId, offset)) {
                    triplet[triplet[44]] = ascManifold[desManifold[neighborId]];
                    triplet[44]++;
                }
                
            }
            sortAndRemoveDuplicates(triplet);
        }

        

        

        int edgesInEG = 0;
        for(int i = 0; i < saddlesLocalToGlobal.size(); i++) {
            auto &triplet = saddleTriplets[i];
            edgesInEG += triplet[44];
        }
        
        return;
    }

    void findDesPaths(  std::vector<std::array<int, 46>> &saddleTriplets,
                        std::vector<int> &maximaLocalToGlobal,
                        std::vector<int> &saddlesLocalToGlobal,
                        const int *saddles2, const int *maximum,
                        const double *offset,
                        const int *desManifold, const int *ascManifold, int &counter_tmp){
        
        
        
        for(int i = 0; i < saddlesLocalToGlobal.size(); i++) {
            const int vertexId = saddles2[i];
            
            auto &triplet = saddleTriplets[i];
            
            triplet[44] = 0;
            
            for(int k = 0; k < maxNeighbors; k++)
            {
                const int neighborId = adjacency[maxNeighbors * vertexId + k];
                
                if(neighborId == -1) continue;
                if(isless(neighborId, vertexId, offset)) {
                    triplet[triplet[44]] = ascManifold[desManifold[neighborId]];
                    triplet[44]++;
                }
                
            }
            sortAndRemoveDuplicates(triplet);
        }

        

        

        int edgesInEG = 0;
        for(int i = 0; i < saddlesLocalToGlobal.size(); i++) {
            auto &triplet = saddleTriplets[i];
            edgesInEG += triplet[44];
        }
        
        return;
    }


    int computeNumberOfLinkComponents(const int* linkVertices, const int nLinkVertices){
        std::unordered_map<int, int> linkVerticesMap;
        for(int i = 0; i < nLinkVertices; i++) {
            const int v = linkVertices[i];
            linkVerticesMap.insert({v, v});
        }

        for(int i = 0; i < nLinkVertices; i++) {
            const int vId = linkVertices[i];

            for(int n = 0; n < maxNeighbors; n++) {
                int uId = adjacency[maxNeighbors * vId + n];
                
                if(uId == -1) continue;
                // // only consider edges in one direction
                

                // only consider edges that are part of the link
                auto u = linkVerticesMap.find(uId);
                if(u == linkVerticesMap.end())
                    continue;

                auto v = linkVerticesMap.find(vId);

                // find
                while(u->first != u->second) {
                    u = linkVerticesMap.find(u->second);
                }
                while(v->first != v->second) {
                    v = linkVerticesMap.find(v->second);
                }
                // union
                u->second = v->first;
            }
        }


        // count components
        int nComponents = 0;
        for(auto kv : linkVerticesMap)
            if(kv.first == kv.second)
                nComponents++;

        return nComponents;
    }
    int classifyVertex(const int vertexId, const double *offset, 
                       const int *desManifold, const int *ascManifold, int *types){
        types[0] = -1;
        types[1] = -1;
        int *upperlinks = new int[maxNeighbors];
        int *lowerlinks = new int[maxNeighbors];

        
        std::array<int, 32> lowerMem; // room for max 32 vertices
        std::array<int, 32> upperMem; // room for max 32 vertices
        std::array<int, 45> lowerMem2; // used by general case
        std::array<int, 45> upperMem2; // used by general case

        
        // lowerMem.fill(-1); 
        // upperMem.fill(-1); 
        lowerMem[0] = -1;
        upperMem[0] = -1;

        int lowerCursor = 0;
        int upperCursor = 0;
        int lowerCursor2 = 0;
        int upperCursor2 = 0;
        
        for(int n = 0; n < maxNeighbors; n++) {
            int u = adjacency[maxNeighbors * vertexId + n];
            if(u == -1) continue;

            if(isless(vertexId, u, offset)) {
                upperMem2[upperCursor2++] = u;
                // if(vertexId == 59962) std::cout<<"neighborId "<<u<<": "<<desManifold[u]<<std::endl;
                // if(std::find(upperMem.begin(), upperMem.end(), desManifold[u]) == upperMem.end()) {
                if(upperMem[upperCursor] != desManifold[u]) {
                    
                    upperMem[++upperCursor] = desManifold[u];
                }
            } else if(islarger(vertexId, u, offset)){
                lowerMem2[lowerCursor2++] = u;
                // if(std::find(lowerMem.begin(), lowerMem.end(), ascManifold[u]) == lowerMem.end()) {
                if(lowerMem[lowerCursor] != ascManifold[u]) {
                    
                    lowerMem[++lowerCursor] = ascManifold[u];
                }
            }
        }

        // if(vertexId == 60415){
        //     for(int i = 0; i<upperCursor2; i++) std::cout<<upperMem2[i]<<", ";
        //     std::cout<<std::endl;
        //     std::cout<<upperCursor<<std::endl;
            
            
        //     for(int i = 0; i<lowerCursor2; i++) std::cout<<lowerMem2[i]<<", ";
        //     std::cout<<std::endl;
        //     std::cout<<lowerCursor<<std::endl;
            

        //     for(int i =0; i<upperCursor+1; i++) std::cout<<upperMem[i]<<", ";
        //     std::cout<<std::endl;

        //     for(int i =0; i<lowerCursor+1; i++) std::cout<<lowerMem[i]<<", ";
        //     std::cout<<std::endl;
        // }

        if(lowerCursor == 0)
        {
            types[0] = 0;
            return 0;
        }

        else if(upperCursor == 0)
        {
            types[0] = 4;
            return 4;
        }
        
        else if(upperCursor == 1 && lowerCursor ==1 )
        {
            types[0] = 5;
            return 5;
        }
        
        if(computeNumberOfLinkComponents(upperMem2.data(), upperCursor2)> 1)
        {
            if(types[0] == -1) types[0] = 2;
            else types[1] = 2;
        }
            
        
        if(computeNumberOfLinkComponents(lowerMem2.data(), lowerCursor2) > 1)
        {
            if(types[0] == -1) types[0] = 1;
            else types[1] = 1;
        }
        

        return 5;
    }

    void getSortedIndexes(const double* vec, int* indexes) {
        
        for (int i = 0; i < num_Elements; ++i) {
            indexes[i] = i;
        }

        
        std::sort(indexes, indexes + num_Elements, [&vec](int i, int j) {
            return vec[i] < vec[j] || (std::abs(vec[i] -  vec[j]) == 0 && i < j); 
        });

        return;
    }

    int* getSortedPositions(const double* arr, int n) {
            int* indices = new int[n];    
            int* positions = new int[n];  

            
            for (int i = 0; i < n; ++i) {
                indices[i] = i;
            }

            
            std::sort(indices, indices + n, [&arr](int i, int j) {
                return arr[i] < arr[j]; 
            });

            
            for (int i = 0; i < n; ++i) {
                positions[indices[i]] = i;
            }

            delete[] indices; 
            return positions;
    }

    int constructMergeTree(std::vector<Branch> &branches,
                            std::vector<std::pair<int, int>>&maximaTriplets, // maximaTriplets[max] = (saddle, largestMax)
                            const std::vector<int> &maximaLocalToGlobal,
                            const std::vector<int> &saddlesLocalToGlobal,
                            const int* order) {
            
            maximaTriplets[maximaTriplets.size() - 1].second
                = maximaTriplets.size() - 1;

            // compress the maximaTriplets
            bool same = false;
            while(!same) {
                same = true;
                // maximaTriplets records the saddle - max connection;
                for(size_t i = 0; i < maximaTriplets.size(); i++) {
                    // find current max's next branch;
                    auto nextBranch = maximaTriplets[maximaTriplets[i].second];
                    // if those two branch's connected max is different
                    // && nextBranch's saddle is lager than current saddle
                    
                    if(nextBranch.second != maximaTriplets[i].second
                        && nextBranch.first < maximaTriplets[i].first) { // we need to follow along larger
                        // saddles to the largestMax
                        maximaTriplets[i].second = nextBranch.second;
                        same = false;
                    }
                }
            }
            
            for(int b = 0; b < (int)maximaTriplets.size();b++) {
                auto &triplet = maximaTriplets[b];
                auto &branch = branches[b];
                auto branchMaxId = maximaLocalToGlobal[b];
                branch.vertices.emplace_back(order[branchMaxId], branchMaxId);

                auto parent = triplet.second;
                if(parent != b) {
                    auto saddle = saddlesLocalToGlobal[triplet.first];
                    auto orderForSaddle = order[saddle];
                    branch.vertices.emplace_back(orderForSaddle, saddle);
                    branch.parentBranch
                        = &branches[parent]; // branches[mainBranch].parentBranch
                                            // = branches[mainBranch];
                    branches[parent].vertices.emplace_back(orderForSaddle, saddle);
                } else {
                    branch.vertices.emplace_back(
                        -1, triplet.first); // triplet.first == globalMin
                }
            }
            for(size_t i = 0; i < branches.size(); i++) {
                auto vect = &branches[i].vertices;
                std::sort(vect->begin(), vect->end(), std::greater<>());
            }

      return 1;
    }

    void computeCP(std::vector<std::pair<int, int>> &maximaTriplets, std::vector<std::pair<int, int>> &minimumTriplets, 
                    std::vector<std::array<int, 46>> &saddleTriplets, std::vector<std::array<int, 46>> &saddle1Triplets, 
                    const double *offset, const int *desManifold, const int *ascManifold,
                    const int *vertex_type_tmp, std::vector<int> &saddles2, 
                    std::vector<int> &saddles1, std::vector<int> &minimum,
                    std::vector<int> &maximum, int &globalMin, int translation){
        

        const int data_size = width * height * depth;
        int nSaddle2 = 0;
        int nMax = 0;
        int nSaddle1 = 0;
        int nMin = 0;

        maximum.clear();
        minimum.clear();
        saddles2.clear();
        saddles1.clear();

        #pragma omp parallel reduction(+:nSaddle2, nMax, nSaddle1)
        {
            std::vector<int> local_saddles2, local_saddles1, local_maximum, local_minimum;

            #pragma omp for
            for(int VertexId = 0; VertexId < data_size; VertexId++) {

                if(vertex_type_tmp[VertexId] == 2 || vertex_type_tmp[VertexId] == 3) {
                    nSaddle2++;
                    local_saddles2.push_back(VertexId);
                }

                if(vertex_type_tmp[VertexId] == 1 || vertex_type_tmp[VertexId] == 3) {
                    nSaddle1++;
                    local_saddles1.push_back(VertexId);
                }

                if(vertex_type_tmp[VertexId] == 4) {
                    nMax++;
                    local_maximum.push_back(VertexId);
                }

                if(vertex_type_tmp[VertexId] == 0) {
                    nMin++;
                    local_minimum.push_back(VertexId);
                }
            }

            #pragma omp critical
            {
                saddles2.insert(saddles2.end(), local_saddles2.begin(), local_saddles2.end());
                saddles1.insert(saddles1.end(), local_saddles1.begin(), local_saddles1.end());
                maximum.insert(maximum.end(), local_maximum.begin(), local_maximum.end());
                minimum.insert(minimum.end(), local_minimum.begin(), local_minimum.end());
            }
        }
        
        
        std::sort(saddles2.begin(), saddles2.end(), [&offset](int v, int u) {
            return offset[v] > offset[u] || ((std::abs(offset[v] - offset[u])) == 0 && v > u);
        });

        std::sort(saddles1.begin(), saddles1.end(), [&offset](int v, int u) {
            return offset[v] < offset[u] || ((std::abs(offset[v] - offset[u])) == 0 && v < u);
        });

        std::sort(maximum.begin(), maximum.end(), [&offset](int v, int u) {
            return offset[v] < offset[u] || ((std::abs(offset[v] - offset[u])) == 0 && v < u);
        });

        // minimum is ranking as descending order, first is the largest minimum
        // need to find saddle1's recheable minimum, and recheable minimum ranking ascending.
        std::sort(minimum.begin(), minimum.end(), [&offset](int v, int u) {
            return offset[v] > offset[u] || ((std::abs(offset[v] - offset[u])) == 0 && v > u);
        });

        saddleTriplets.clear();
        saddleTriplets.resize(nSaddle2);

        saddle1Triplets.clear();
        saddle1Triplets.resize(nSaddle1);

        nMax = maximum.size();
        nMin = minimum.size();

        int *tempArray = new int[data_size];
        for(int i = 0; i < nMax; i++) {
            tempArray[maximum[i]] = i;
        }

        int *tempArray1 = new int[data_size];
        for(int i = 0; i < nMin; i++) {
            tempArray1[minimum[i]] = i;
        }

        int counter_tmp = 0;
        findAscPaths(saddleTriplets, maximum, saddles2, saddles2.data(), maximum.data(), offset,
                    desManifold, tempArray, counter_tmp);

        counter_tmp = 0;
        findDesPaths(saddle1Triplets, minimum, saddles1, saddles1.data(), minimum.data(), offset,
                    ascManifold, tempArray1, counter_tmp);

        
        int q = 0;
        for(auto &item:saddleTriplets){
            item[45] = saddles2[q++];
        }

        q = 0;
        for(auto &item:saddle1Triplets){
            item[45] = saddles1[q++];
        }
            
        if(translation == 0) return;
    
        int *sortedVertex = new int[data_size];
        getSortedIndexes(offset, sortedVertex);
        globalMin = sortedVertex[0];
        
        return; 
    }

    // void computeCP(std::vector<std::pair<int, int>> &maximaTriplets, std::vector<std::array<int, 46>> &saddleTriplets, 
    //                 const double *offset, const int *desManifold, const int *vertex_type_tmp, std::vector<int> &saddles2, 
    //                 std::vector<int> &maximum, int &globalMin, int translation){
        

    //     const int data_size = width * height * depth;
    //     int nSaddle2 = 0;
    //     int nMax = 0;

    //     maximum.clear();
    //     saddles2.clear();

    //     std::vector<int> saddles1;

    //     int nSaddle1 = 0;
    //     #pragma omp parallel reduction(+:nSaddle2, nMax, nSaddle1)
    //     {
    //         std::vector<int> local_saddles2, local_saddles1, local_maximum, local_minimum;

    //         #pragma omp for
    //         for(int VertexId = 0; VertexId < data_size; VertexId++) {
                
    //             if(vertex_type_tmp[VertexId] == 2 || vertex_type_tmp[VertexId] == 3) {
    //                 nSaddle2++;
    //                 local_saddles2.push_back(VertexId);
    //             }

    //             if(vertex_type_tmp[VertexId] == 1 || vertex_type_tmp[VertexId] == 3) {
    //                 nSaddle1++;
    //                 local_saddles1.push_back(VertexId);
    //             }

    //             if(vertex_type_tmp[VertexId] == 4) {
    //                 nMax++;
    //                 local_maximum.push_back(VertexId);
    //             }

    //         }

            
    //         #pragma omp critical
    //         {
    //             saddles2.insert(saddles2.end(), local_saddles2.begin(), local_saddles2.end());
    //             saddles1.insert(saddles1.end(), local_saddles1.begin(), local_saddles1.end());
    //             maximum.insert(maximum.end(), local_maximum.begin(), local_maximum.end());
                
    //         }
    //     }

    //     std::sort(saddles2.begin(), saddles2.end(), [&offset](int v, int u) {
    //         return offset[v] > offset[u] || ((std::abs(offset[v] - offset[u])) == 0 && v > u);
    //     });

    //     std::sort(maximum.begin(), maximum.end(), [&offset](int v, int u) {
    //         return offset[v] < offset[u] || ((std::abs(offset[v] - offset[u])) == 0 && v < u);
    //     });

    //     saddleTriplets.clear();
    //     saddleTriplets.resize(nSaddle2);
        

    //     int *tempArray = new int[data_size];
    //     for(int i = 0; i < nMax; i++) {
    //         tempArray[maximum[i]] = i;
            
    //     }

    //     int counter_tmp = 0;
    //     findAscPaths(saddleTriplets, maximum, saddles2, saddles2.data(), maximum.data(), offset,
    //                 desManifold, tempArray, counter_tmp);

        
    //     int q = 0;
    //     for(auto &item:saddleTriplets)
    //     {
    //         item[45] = saddles2[q++];
    //     }
            
    //     if(translation == 0) return;
        

    //     int *sortedVertex = new int[data_size];
    //     getSortedIndexes(offset, sortedVertex);
    //     globalMin = sortedVertex[0];

        

    //     return; 
    // }

    int constructPersistencePairs(std::vector<std::pair<int, int>> &pairs,
                                    std::vector<std::pair<int, int>> &maximaTriplets,
                                    std::vector<std::array<int, 46>> &saddleTriplets,
                                    const int* Maximum,const int* saddles2,const int nMax,int globalMin) {
        
        int step = 0;
        bool changed = true;
        
        std::vector<int> maximumPointer(nMax);
        std::iota(std::begin(maximumPointer), std::end(maximumPointer), 0);
        int globalMax = nMax - 1;

        while(changed) {
            
            changed = false;

            std::vector<int> largestSaddlesForMax(nMax, saddleTriplets.size());
            // saddle is stored by descending rank;
            // find the largest saddle connected with eahc max;
            for(int i = 0; i < (int)saddleTriplets.size(); i++) {
                auto &triplet = saddleTriplets[i];
                int temp;
                for(int p = 0; p < triplet[44]; p++) {
                    
                    const auto &max = triplet[p];
                    if(max != globalMax) {
                        temp = largestSaddlesForMax[max];
                        if(i < temp) {
                        // save only maximum saddle
                        // smaller id -> larger saddles
                            largestSaddlesForMax[max]
                                = std::min(i, largestSaddlesForMax[max]);
                        }
                    }
                }
            }
            
            std::vector<int> lActiveMaxima;
            lActiveMaxima.reserve(nMax);

            for(size_t i = 0; i < largestSaddlesForMax.size(); i++) {
                int maximum = i;
                auto largestSaddle = largestSaddlesForMax[maximum];
                if(largestSaddle < (int)saddleTriplets.size()) {
                    // check if this maximum is also the smallest max conencted with this largestSaddle;
                    const auto &triplet = saddleTriplets[largestSaddle];
                    if(triplet[0] == maximum) { 
                        // smallest maximum reachable from the saddle
                        // only connect the smallest maximum with the saddle;
                        // will the saddle is the largest neighbor of this maximum;
                        changed = true;
                        pairs[maximum] = std::make_pair(saddles2[largestSaddle], Maximum[maximum]);
                        auto largestMax = triplet[triplet[44] - 1];
                        maximumPointer[maximum] = largestMax;

                        maximaTriplets[maximum] = (std::make_pair(largestSaddle, largestMax));
                        lActiveMaxima.push_back(maximum);
                        
                    }
                }
            }
            // use pathcompression on the maximumPointer
            size_t lnActiveMaxima = lActiveMaxima.size();
            size_t currentIndex = 0;

            while(lnActiveMaxima > 0) {
                for(size_t i = 0; i < lnActiveMaxima; i++) {
                    int &v = lActiveMaxima[i];
                    int &nextPointer = maximumPointer[v];
                    nextPointer = maximumPointer[nextPointer];
                    if(nextPointer != maximumPointer[nextPointer]) {
                        lActiveMaxima[currentIndex++] = v;
                    }
                }
                lnActiveMaxima = currentIndex;
                currentIndex = 0;
            }

            for(size_t i = 0; i < saddleTriplets.size(); i++) {
                auto &triplet = saddleTriplets[i];
                for(int r = 0; r < triplet[44]; r++) {
                    triplet[r] = maximumPointer[triplet[r]];
                }
                sortAndRemoveDuplicates(triplet);
                if(triplet[44] == 1) {
                    triplet[44] = 0;
                }
            }
            step++;
        }

        
        // the global max is always in the last position of the
        // maximaLocalToGlobal vector and needs to connect with the global
        // minimum
        pairs[pairs.size() - 1] = std::make_pair(globalMin, Maximum[nMax - 1]);
        maximaTriplets[maximaTriplets.size() - 1] = std::make_pair(globalMin, maximaTriplets.size() - 1); // maximaTriplets[globalmax] = (globalmin, globalmax);

        // int counter = 0;
        // for(auto &item:maximaTriplets)
        // {
        //     if(counter != maximaTriplets.size())
        //     {
        //         item.first = saddles2[item.first];
        //         item.second = Maximum[item.second];
        //         counter++;
        //     }
        //     else
        //     {
        //         item.second = Maximum[item.second];
        //         counter++;
        //     }
            
        // }

        
        return 1;
    }

    int computeMaxLabel(int i, const double *offset){
        
        int current_id = i;
        int largest_neighbor = current_id;  

        while (true) {
            int next_largest_neighbor = largest_neighbor;
            
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

    int computeMinLabel(int i, const double *offset){
        
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

    void computelargestSaddlesForMax(const int nMax, 
                                    const std::vector<std::array<int, 46>> &saddleTriplets, 
                                    std::unordered_map<int, int> &largestSaddlesForMax,
                                    const std::vector<int> *maximum){
        // saddle is stored by descending rank;
        // find the largest saddle connected with eahc max;
        largestSaddlesForMax.clear();
        for(int i = 0; i<nMax; i++){
            largestSaddlesForMax[(*maximum)[i]] = saddleTriplets.size();
        }

        
        int globalMax = nMax - 1;
        
        for(int i = 0; i < (int)saddleTriplets.size(); i++) {
            auto &triplet = saddleTriplets[i];
            int temp;
            for(int p = 0; p < triplet[44]; p++) {
               
                const auto &max = triplet[p];
                
                if(max != globalMax) {
                    temp = largestSaddlesForMax[(*maximum)[max]];
                    if(i < temp) {
                        largestSaddlesForMax[(*maximum)[max]]
                            = std::min(i, largestSaddlesForMax[(*maximum)[max]]);
                    }
                }
            }
        }
    }

    void computesmallestSaddlesForMin(const int nMin, 
                                    const std::vector<std::array<int, 46>> &saddleTriplets, 
                                    std::unordered_map<int, int> &smallestSaddlesForMin,
                                    const std::vector<int> *minimum){
        // saddle is stored by descending rank;
        // find the smallest saddle connected with eahc min;
        smallestSaddlesForMin.clear();
        for(int i = 0; i < nMin; i++){
            smallestSaddlesForMin[(*minimum)[i]] = saddleTriplets.size();
        }

        
        int globalMin = nMin - 1;
        
        for(int i = 0; i < (int)saddleTriplets.size(); i++) {
            auto &triplet = saddleTriplets[i];
            int temp;
            for(int p = 0; p < triplet[44]; p++) {
               
                const auto &min = triplet[p];
                
                if(min != globalMin) {
                    temp = smallestSaddlesForMin[(*minimum)[min]];
                    if(i < temp) {
                        smallestSaddlesForMin[(*minimum)[min]]
                            = std::min(i, smallestSaddlesForMin[(*minimum)[min]]);
                    }
                }
            }
        }
    }

    void compute_Max_for_Saddle(int saddle, const double *offset){
        int label_count = 0;
        for(int j = 0; j< maxNeighbors; j++)
        {
            int neighborId = adjacency[ maxNeighbors * saddle + j];
            if(neighborId == -1) continue;
            if(islarger(neighborId, saddle, offset)){
                int l = computeMaxLabel(neighborId, offset);
                or_saddle_max_map[saddle * 4 + label_count] = l;
                label_count++;
            }

        }
        return;
    }

    void compute_Min_for_Saddle(int saddle, const double *offset){
        int label_count = 0;
        for(int j = 0; j< maxNeighbors; j++)
        {
            int neighborId = adjacency[ maxNeighbors * saddle + j];
            if(neighborId == -1) continue;
            if(isless(neighborId, saddle, offset)){
                int l = computeMinLabel(neighborId, offset);
                or_saddle_min_map[saddle * 4 + label_count] = l;
                label_count++;
            }

        }
        return;
    }

    void get_wrong_split_neighbors(int saddle, int &num_false_cases, const double *offset){
        int label_count = 0;
        for(int j = 0; j< maxNeighbors; j++)
        {
            int neighborId = adjacency[ maxNeighbors * saddle + j];
            if(neighborId == -1) continue;
            if(islarger(neighborId, saddle, offset)){
                
                int l = computeMaxLabel(neighborId, offset);
                if(l!=or_saddle_max_map[saddle * 4 + label_count]){
                    if(wrong_neighbors[neighborId] == 0){
                        wrong_neighbors[neighborId] = 1;
                        wrong_neighbors_index[num_false_cases] = neighborId;
                        num_false_cases++;
                        return;
                    }
                }
                label_count++;
            }

        }
    }

    void get_wrong_join_neighbors(int saddle, int &num_false_cases, const double *offset){
        int label_count = 0;
        for(int j = 0; j< maxNeighbors; j++)
        {
            int neighborId = adjacency[ maxNeighbors * saddle + j];
            if(neighborId == -1) continue;

            if(isless(neighborId, saddle, offset)){
                
                int l = computeMinLabel(neighborId, offset);
                if(l!=or_saddle_min_map[saddle * 4 + label_count]){
                    if(wrong_neighbors_ds[neighborId] == 0){
                        wrong_neighbors_ds[neighborId] = 1;
                        wrong_neighbors_ds_index[num_false_cases] = neighborId;
                        num_false_cases++;
                        return;
                    }
                }
                label_count++;
            }

        }
    }

    int fixpath(int i, int direction = 0, std::string type = "split"){
        double delta;
        int true_index = -1;
        int false_index = -1;
        if(direction == 0){
            
            int current_id = i;
            int largest_neighbor = current_id;  
            
            while (true) {
                int next_largest_neighbor = largest_neighbor;
                int dec_next_largest_neighbor = largest_neighbor;
                
                for (int j = 0; j < maxNeighbors; j++) {
                    int neighbor_id = adjacency[maxNeighbors * current_id + j];
                    if (neighbor_id == -1) continue;  
                    
                    if (input_data[next_largest_neighbor] < input_data[neighbor_id] || 
                    (input_data[next_largest_neighbor] == input_data[neighbor_id] && next_largest_neighbor < neighbor_id)) {
                        next_largest_neighbor = neighbor_id;
                    }
                    if (decp_data[dec_next_largest_neighbor] < decp_data[neighbor_id] || 
                    (decp_data[dec_next_largest_neighbor] == decp_data[neighbor_id] && dec_next_largest_neighbor < neighbor_id)) {
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
            
            
            if(false_index==true_index){
                
                return 0;
            }
            
            if(type == "split"){
                double d = ((input_data[false_index] - bound) + decp_data[false_index]) / 2.0 - decp_data[false_index];
            
                double oldValue = d_deltaBuffer[false_index];
                if (d > oldValue) {
                    swap(false_index, d);
                } 
            }
            else{
                double d = ((input_data[true_index] + bound) + decp_data[true_index]) / 2.0 - decp_data[true_index];
            
                double oldValue = d_deltaBuffer[true_index];
                if (d > oldValue) {
                    swap(true_index, d);
                } 
            }
             

            return 0;
            
        }

        else {
            
            int current_id = i;
            int largest_neighbor = current_id;  
            
            while (true) {
                int next_largest_neighbor = largest_neighbor;
                int dec_next_largest_neighbor = largest_neighbor;
                
                for (int j = 0; j < maxNeighbors; j++) {
                    int neighbor_id = adjacency[maxNeighbors * current_id + j];
                    if (neighbor_id == -1) continue;  
                    
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

            if(false_index==true_index) return 0;

            if(type == "split"){
                double d = ((input_data[true_index] - bound) + decp_data[true_index]) / 2.0 - decp_data[true_index];
                double oldValue = d_deltaBuffer[true_index];
                if (d > oldValue) {
                    swap(true_index, d);
                } 
            }
            else{
                double d = ((input_data[false_index] + bound) + decp_data[false_index]) / 2.0 - decp_data[false_index];
                double oldValue = d_deltaBuffer[false_index];
                if (d > oldValue) {
                    swap(false_index, d);
                } 
            }
             

            return 0;
            
        
        }
        return 0;
    }

    void r_loops(std::string type = "split"){
        
        init_delta();
        for(int i = 0;i< number_of_false_cases; i++)
        {
            int id = wrong_neighbors_index[i];
            fixpath(id, 0, type);
        }

        for(int i = 0;i< number_of_false_cases1; i++)
        {
            
            int id = wrong_neighbors_ds_index[i];
            fixpath(id, 1, type);
        }
        apply_delta(type);
    }

    void get_wrong_index_max(){
        wrong_max_counter = 0;
        wrong_max_counter_2 = 0;
        std::fill(wrong_rank_max, wrong_rank_max + num_Elements, 0);
        std::fill(wrong_rank_max_2, wrong_rank_max_2 + num_Elements, 0);

        for(int i = 0; i<dec_saddleTriplets.size(); i++){

            if(maximum[saddleTriplets[i][0]] != dec_maximum[dec_saddleTriplets[i][0]]){
                
                int maxId = maximum[saddleTriplets[i][0]];
                if(wrong_rank_max[maxId] == 0)
                {
                    wrong_rank_max_index[wrong_max_counter * 2] = maximum[saddleTriplets[i][0]];
                    wrong_rank_max_index[wrong_max_counter * 2 + 1] = dec_maximum[dec_saddleTriplets[i][0]];
                    wrong_rank_max[maxId] = 1;

                    wrong_max_counter ++;
                    
                }
            }

            if(maximum[saddleTriplets[i][saddleTriplets[i][44] - 1]] != dec_maximum[dec_saddleTriplets[i][dec_saddleTriplets[i][44] - 1]]){
                int maxId = maximum[saddleTriplets[i][saddleTriplets[i][44] - 1]];
                if(wrong_rank_max_2[maxId] == 0){
                    wrong_rank_max_index_2[wrong_max_counter_2 * 2] = maximum[saddleTriplets[i][saddleTriplets[i][44] - 1]];
                    wrong_rank_max_index_2[wrong_max_counter_2 * 2 + 1] = dec_maximum[dec_saddleTriplets[i][dec_saddleTriplets[i][44] - 1]];
                    wrong_rank_max_2[maxId] = 1;
                    
                    wrong_max_counter_2 ++;
                    
                }
            }
            
        }
    }

    void get_wrong_index_min(){
        wrong_min_counter = 0;
        wrong_min_counter_2 = 0;
        std::fill(wrong_rank_min, wrong_rank_min + num_Elements, 0);
        std::fill(wrong_rank_min_2, wrong_rank_min_2 + num_Elements, 0);

        
        for(int i = 0; i<dec_saddle1Triplets.size(); i++)
        {
            // find each saddle's distorted rechable largest minimum
            
            if(minimum[saddle1Triplets[i][0]] != dec_minimum[dec_saddle1Triplets[i][0]]){
                
                int minId = minimum[saddle1Triplets[i][0]];
                if(wrong_rank_min_2[minId] == 0){
                    wrong_rank_min_index_2[wrong_min_counter_2 * 2] = minimum[saddle1Triplets[i][0]];
                    wrong_rank_min_index_2[wrong_min_counter_2 * 2 + 1] = dec_minimum[dec_saddle1Triplets[i][0]];
                    wrong_rank_min_2[minId] = 1;
                    
                    wrong_min_counter_2 ++;
                    
                }
            }

            // find each saddle's distorted rechable smallest minimum
            if(minimum[saddle1Triplets[i][saddle1Triplets[i][44] - 1]] != dec_minimum[dec_saddle1Triplets[i][dec_saddle1Triplets[i][44] - 1]]){
                int minId = minimum[saddle1Triplets[i][saddle1Triplets[i][44] - 1]];
                if(wrong_rank_min[minId] == 0){
                    wrong_rank_min_index[wrong_min_counter * 2] = minimum[saddle1Triplets[i][saddle1Triplets[i][44] - 1]];
                    wrong_rank_min_index[wrong_min_counter * 2 + 1] = dec_minimum[dec_saddle1Triplets[i][dec_saddle1Triplets[i][44] - 1]];
                    wrong_rank_min[minId] = 1;

                    wrong_min_counter ++;
                    
                }
            }
            
        }
    }

    void get_wrong_index_saddles(){
        wrong_saddle_counter = 0;

        std::fill(wrong_rank_saddle, wrong_rank_saddle + num_Elements, 0);
        
        for (const auto& pair : dec_largestSaddlesForMax) {
            const int maxId = pair.first;
            if(dec_saddles2[pair.second] != saddles2[largestSaddlesForMax[maxId]]){
                if(wrong_rank_saddle[maxId] == 0){
                    
                    wrong_rank_saddle_index[wrong_saddle_counter * 2] = saddles2[largestSaddlesForMax[maxId]];
                    wrong_rank_saddle_index[wrong_saddle_counter * 2 + 1] = dec_saddles2[dec_largestSaddlesForMax[maxId]];
                    int i = maxId;
                    
                    wrong_rank_saddle[maxId] = 1;
                    wrong_saddle_counter ++;
                    std::cout<<wrong_rank_saddle<<std::endl;
                }
            }
        }

    }

    void get_wrong_index_saddles_join(){
        wrong_saddle_counter_join = 0;

        std::fill(wrong_rank_saddle_join, wrong_rank_saddle_join + num_Elements, 0);
        
        for (const auto& pair : dec_smallestSaddlesForMin) {
            const int minId = pair.first;
            if(dec_saddles1[pair.second] != saddles1[smallestSaddlesForMin[minId]]){
                if(wrong_rank_saddle_join[minId] == 0){
                    
                    wrong_rank_saddle_join_index[wrong_saddle_counter_join * 2] = saddles1[smallestSaddlesForMin[minId]];
                    wrong_rank_saddle_join_index[wrong_saddle_counter_join * 2 + 1] = dec_saddles1[dec_smallestSaddlesForMin[minId]];
                    int i = minId;
                    
                    wrong_rank_saddle_join[minId] = 1;
                    wrong_saddle_counter_join ++;
                }
            }
        }

    }

    void fix_wrong_index_max(int true_index, int false_index, int direction = 0, std::string type = "split"){
        double tmp_delta;
        double tmp_true_value = decp_data[true_index];
        double tmp_false_value = decp_data[false_index];

        if(type == "split"){
            if(direction == 0){
            
                tmp_true_value = (input_data[true_index] - bound + decp_data[true_index]) / 2.0;
                double d = tmp_true_value - decp_data[true_index];
            
                double oldValue = d_deltaBuffer[true_index];
                
                if (d > oldValue) {
                    swap(true_index, d);
                }  
            }

            else if(direction == 1){
                
                tmp_false_value = (input_data[false_index] - bound + decp_data[false_index]) / 2.0;
                double d = tmp_false_value - decp_data[false_index];
            
                double oldValue = d_deltaBuffer[false_index];
                
                if (d > oldValue) {
                    swap(false_index, d);
                } 
            }
        }

        else{
            if(direction == 0){
            
                tmp_false_value = (input_data[false_index] + bound + decp_data[false_index]) / 2.0;
                double d = tmp_false_value - decp_data[false_index];
            
                double oldValue = d_deltaBuffer[false_index];
                
                if (d > oldValue) {
                    swap(false_index, d);
                }  
            }

            else if(direction == 1){
                
                tmp_true_value = (input_data[true_index] + bound + decp_data[true_index]) / 2.0;
                double d = tmp_true_value - decp_data[true_index];
            
                double oldValue = d_deltaBuffer[true_index];
                
                if (d > oldValue) {
                    swap(true_index, d);
                } 
            }
        }

        return;
    }

    void fix_wrong_index_min(int true_index, int false_index, int direction = 0, std::string type = "split"){
        double tmp_delta;
        double tmp_true_value = decp_data[true_index];
        double tmp_false_value = decp_data[false_index];

        
        if(direction == 0){
        
            tmp_true_value = (input_data[true_index] - bound + decp_data[true_index]) / 2.0;
            double d = tmp_true_value - decp_data[true_index];
        
            double oldValue = d_deltaBuffer[true_index];
            
            if (d > oldValue) {
                swap(true_index, d);
            }  
        }

        else if(direction == 1){
            
            tmp_false_value = (input_data[false_index] - bound + decp_data[false_index]) / 2.0;
            double d = tmp_false_value - decp_data[false_index];
        
            double oldValue = d_deltaBuffer[false_index];
            
            if (d > oldValue) {
                swap(false_index, d);
            } 
        }
        

        return;
    }

    void fix_wrong_index_saddle(int true_index, int false_index, int direction = 0, std::string type = "split"){
        double tmp_delta;
        double tmp_true_value = decp_data[true_index];
        double tmp_false_value = decp_data[false_index];
        if(tmp_true_value > tmp_false_value || (tmp_true_value == tmp_false_value && true_index > false_index)) return;
        if(direction == 0){
            double d = (input_data[false_index] - bound + decp_data[false_index])/2.0 - decp_data[false_index];
            double oldValue = d_deltaBuffer[false_index];
            if (d > oldValue) {
                swap(false_index, d);
            }  
        }

        else{
            
            double d = (input_data[true_index] + bound + decp_data[true_index])/2.0 - decp_data[true_index];
            double oldValue = d_deltaBuffer[true_index];
            if (d > oldValue) {
                swap(true_index, d);
            }  
        
        }
        

        return;
    }

    void s_loops(std::string type = "split"){
        get_wrong_index_max();
        init_delta();

        for(int i = 0; i < wrong_max_counter; i++)
        {
            
            int true_index = wrong_rank_max_index[i*2];
            int false_index = wrong_rank_max_index[i*2+1];
            
            fix_wrong_index_max(true_index, false_index, 0, type);
        }

        for(int i = 0; i < wrong_max_counter_2; i++)
        {
            
            int true_index = wrong_rank_max_index_2[i*2];
            int false_index = wrong_rank_max_index_2[i*2+1];
            
            fix_wrong_index_max(true_index, false_index, 1, type);
        }

        apply_delta(type);

    }

    void s_loops_join(std::string type = "split"){
        get_wrong_index_min();
        init_delta();
        std::cout<<wrong_min_counter<<", "<<wrong_min_counter_2<<std::endl;
        for(int i = 0; i < wrong_min_counter; i++){
            
            int true_index = wrong_rank_min_index[i*2];
            int false_index = wrong_rank_min_index[i*2+1];
            
            fix_wrong_index_min(true_index, false_index, 0, type);
        }

        for(int i = 0; i < wrong_min_counter_2; i++){
            
            int true_index = wrong_rank_min_index_2[i*2];
            int false_index = wrong_rank_min_index_2[i*2+1];
            
            fix_wrong_index_min(true_index, false_index, 1, type);
        }

        apply_delta(type);

    }

    void saddle_loops(std::string type = "split"){
        get_wrong_index_saddles();
        init_delta();
        // std::cout<<wrong_saddle_counter<<std::endl;
        for(int i = 0; i < wrong_saddle_counter; i++)
        {
            
            int true_index = wrong_rank_saddle_index[i*2];
            int false_index = wrong_rank_saddle_index[i*2+1];
            
            fix_wrong_index_saddle(true_index, false_index, 0, type);
        }
        apply_delta(type);
    }

    void saddle_loops_join(std::string type = "split"){
        get_wrong_index_saddles_join();
        init_delta();

        for(int i = 0; i < wrong_saddle_counter_join; i++)
        {
            
            int true_index = wrong_rank_saddle_join_index[i*2];
            int false_index = wrong_rank_saddle_join_index[i*2+1];
            
            fix_wrong_index_saddle(true_index, false_index, 1, type);
        }
        apply_delta(type);
    }

    void saveArrayToBin(const double* arr, size_t size, const std::string& filename) {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return;
        }

        outFile.write(reinterpret_cast<const char*>(arr), size * sizeof(double));

        outFile.close();
    }

    void compute_MergeT(std::vector<Branch> &branches, 
                        const std::vector<int> saddles2, 
                        const std::vector<int> maximum,
                        std::vector<std::pair<int, int>> &maximaTriplets,
                        std::vector<std::array<int, 46>> &saddleTriplets,
                        const double *offset,
                        const int data_size, const int globalMin){

        int nMax = maximum.size();
        std::vector<std::pair<int, int>> persistencePairs;
        maximaTriplets.resize(nMax);
        persistencePairs.resize(nMax);
        
        constructPersistencePairs(persistencePairs, maximaTriplets, saddleTriplets, maximum.data(), saddles2.data(), nMax, globalMin);


        branches.resize(nMax);

        int* order = getSortedPositions(offset, data_size);
        
        constructMergeTree(branches, maximaTriplets, maximum,
                           saddles2, order);
    }

    void reorderVectorByTemparray(std::vector<std::pair<int, int>>& vec, const std::vector<int>& temparray) {
        
        std::vector<size_t> indices(vec.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
            
        }

        // 按 temparray 的值对 indices 排序
        std::sort(indices.begin(), indices.end(), [&temparray](size_t i1, size_t i2) {
            return temparray[i1] < temparray[i2];
        });

        // 按排序后的 indices 重排 vec
        std::vector<std::pair<int, int>> sorted_vec(vec.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_vec[i] = vec[indices[i]];
        }
        std::cout<<std::endl;
        vec = sorted_vec;
    }
}