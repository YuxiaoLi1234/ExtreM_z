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

extern "C" {
    int directions[78] = {
        1, 0, 0, 
        -1, 0, 0,   
        0, 1, 0, 
        0, -1, 0,
        -1, 1, 0,
        1, -1, 0, 
        0, 0, -1, 0, -1, 1,
        0, 0, 1, 0, 1, -1,
        -1, 0, 1, 1, 0, -1,
        1, 1, 0, -1, -1, 0,
        1, 0, 1, -1, 0, -1,
        0, 1, 1, 0, -1, -1,
        1, 1, 1, 1, 1, -1,  
        1, -1, 1, 1, -1, -1,
        -1, 1, 1, -1, 1, -1,
        -1, -1, 1, -1, -1, -1
    };

    double* d_deltaBuffer;
    int width, height, depth, maxNeighbors, num_Elements;
    int* adjacency;
    double *decp_data, *input_data;
    double bound;
    int *dec_vertex_type, *vertex_type;
    int count_f_max, count_f_min, count_f_saddle, number_of_false_cases, wrong_max_counter, globalMin, dec_globalMin;
    int *all_max, *all_min, *all_saddle;
    std::vector<std::vector<int>> vertex_cells;
    int *or_saddle_max_map, *wrong_neighbors, *wrong_neighbors_index, *wrong_rank_max, *wrong_rank_max_index;
    std::vector<std::array<int, 46>> saddleTriplets, dec_saddleTriplets;
    std::vector<int> saddles2, maximum, dec_saddles2, dec_maximum;

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

    void getdata(const std::string &filename, 
                const double er, 
                int data_size) {
        
        
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::cout<<data_size<<std::endl;
        if (size != static_cast<std::streamsize>(data_size * sizeof(double))) {
            std::cerr << "File size does not match expected data size." << std::endl;
            return;
        }

        file.read(reinterpret_cast<char *>(input_data), size);
        if (!file) {
            std::cerr << "Error reading file." << std::endl;
            return;
        }

        // Step 2: 配置 SZ3 压缩
        SZ3::Config conf(data_size); // 1D 数据，长度为 data_size
        conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
        conf.errorBoundMode = SZ3::EB_REL;
        conf.relErrorBound = er; // 设置绝对误差界限

        // Step 3: 压缩数据
        size_t cmpSize = 0;
        char *compressedData = SZ_compress(conf, input_data, cmpSize);

        // // Step 4: 解压缩数据
        
        
        SZ_decompress(conf, compressedData, cmpSize, decp_data);
        // // Step 5: 释放压缩数据的内存
        delete[] compressedData;

        // 计算误差界限（bound）
        double minValue = *std::min_element(input_data, input_data + data_size);
        double maxValue = *std::max_element(input_data, input_data + data_size);
        bound = (maxValue - minValue) * er;

        std::cout << "Data read, compressed, and decompressed successfully." << std::endl;
    }


    int ComputeDescendingManifold(const double *offset,
                                  int *DS_M){
        #pragma omp parallel for
        for (int i = 0; i < num_Elements; i++) {
            int largest_neighbor = i;
            for(int j = 0; j< maxNeighbors; j++)
            {
                int neighbor_id = adjacency[ maxNeighbors * i + j];
                if(neighbor_id == -1) continue;
                if(offset[largest_neighbor] < offset[neighbor_id] || (offset[largest_neighbor] == offset[neighbor_id] && largest_neighbor < neighbor_id)) largest_neighbor = neighbor_id;
            }

            DS_M[i] = largest_neighbor;
        }

        #pragma omp parallel for
        for (int i = 0; i < num_Elements; i++)
        {
            int current_id = DS_M[i];
            int next_id = DS_M[current_id];
            while(current_id != next_id)
            {
                DS_M[i] = next_id;
                current_id = DS_M[i];
                next_id = DS_M[current_id];
            }
        }

    };

    int ComputeAscendingManifold(const double *offset, 
                                  int *AS_M){
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

        #pragma omp parallel for
        for (int i = 0; i < num_Elements; i++)
        {
            int current_id = AS_M[i];
            int next_id = AS_M[current_id];
            while(current_id != next_id)
            {
                AS_M[i] = next_id;
                current_id = AS_M[i];
                next_id = AS_M[current_id];
            }
        }

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
        return offset[v] < offset[u] || (offset[v] == offset[u] && v < u);
    }

    bool islarger(const int v, const int u, const double *offset){
        return offset[v] > offset[u] || (offset[v] == offset[u] && v > u);
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
    int classifyVertex1(const double* heightMap, int i, int type=0) {
    
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
    
        for (int d =0;d<maxNeighbors;d++) {
            if(adjacency[vertexId*maxNeighbors+d]==-1) continue;
            int r = adjacency[i*maxNeighbors+d];
            
            if (heightMap[r] < currentHeight) {
                lowerStar.emplace_back(r);
            } 
            else if (heightMap[r] == currentHeight and r<i) {
                lowerStar.emplace_back(r);
            }

            if (heightMap[r] > currentHeight) {
                upperStar.emplace_back(r);
            } 
            else if (heightMap[r] == currentHeight and r>i) {
                upperStar.emplace_back(r);
            }

        }
        
        

        std::vector<UnionFind> lowerSeeds(lowerStar.size());
        std::vector<UnionFind> upperSeeds(upperStar.size());
        std::vector<UnionFind *> lowerList(lowerStar.size());
        std::vector<UnionFind *> upperList(upperStar.size());

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
            // cout<<"yes"<<endl;
        // minimum: 0 1-saddle: 1, 2-saddle: 2, regular: 5 maximum: 4;
        if(lowerComponentNumber == 0 && upperComponentNumber == 1) return 0;
        else if(lowerComponentNumber == 1 && upperComponentNumber == 0) return 4;
        else if(lowerComponentNumber == 1 && upperComponentNumber == 1) return 5;
        else if(lowerComponentNumber > 1) return 1;
        else if(upperComponentNumber > 1) return 2;
            

        return 5;
    }

    void classifyAllVertices(int *vertex_type_tmp, const double* heightMap, 
                             int type=0) {
        #pragma omp parallel for
        for (int i = 0; i < num_Elements; ++i) {
            vertex_type_tmp[i] = classifyVertex1(heightMap, i, type);
        }
    }

    void init_delta(){
        #pragma omp parallel for
        for(int tid = 0; tid < num_Elements; tid++)
        {
            d_deltaBuffer[tid] = -2000.0 * bound;
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

    bool atomicCASDouble(double* ptr, double old_val, double new_val) {
        // 将 double 指针转换为 uint64_t 指针
        bool swapped = false;
        #pragma omp critical
        {
            if (*ptr == old_val) {
                *ptr = new_val;
                swapped = true;
            }
        }
        return swapped;
    }


    int swap(int index, double delta){
        int update_successful = 0;
        double oldValue = d_deltaBuffer[index];
        double new_edit = delta;
        double old_edit = d_deltaBuffer[index];
        while (new_edit > old_edit) {
            if (atomicCASDouble(&d_deltaBuffer[index],old_edit, new_edit)) {
                break;
            }
            old_edit = d_deltaBuffer[index];
        }

        return 0;
    }

    void c_loop(int index, int direction = 0)
    {      
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
        

        

        return;
    }

    void get_false_criticle_points(){
    

        count_f_max=0;
        count_f_min=0;
        count_f_saddle=0;

        #pragma omp parallel for reduction(+:count_f_max, count_f_min, count_f_saddle)
        for (auto i = 0; i < num_Elements; i ++) {
            int type1 = dec_vertex_type[i];
            
            if(type1 != vertex_type[i]){
                
                // maximum
                if((type1==1 and vertex_type[i]!=4) or (type1!=1 and vertex_type[i]==4)){
                    
                    all_max[count_f_max] = i;
                    count_f_max+=1;
                } 
                if((type1==-1 and vertex_type[i]!=0) or (type1!=-1 and vertex_type[i]==0)){
                    all_min[count_f_min] = i;
                    count_f_min+=1;
                }
                if((type1==2 and vertex_type[i]!=2) or (type1!=2 and vertex_type[i]==2)){
                    
                    all_saddle[count_f_saddle] = i;
                    count_f_saddle+=1;
                }

                if((type1==1 and vertex_type[i]!=1) or (type1!=1 and vertex_type[i]==2)){
                    
                    all_saddle[count_f_saddle] = i;
                    count_f_saddle+=1;
                }
            }

            else if(type1==2 and vertex_type[i]==2 || type1==1 and vertex_type[i]==1){
                    std::vector<int> lowerStar, upperStar, or_lowerStar, or_upperStar;
                    double currentHeight = decp_data[i];
                    double or_currentHeight = input_data[i];
                    for (int d =0;d<maxNeighbors;d++) {
                        if(adjacency[i*maxNeighbors+d]==-1) continue;
                        int r = adjacency[i*maxNeighbors+d];
                        
                        
                        if (decp_data[r] < currentHeight) {
                            lowerStar.emplace_back(r);
                        } 
                        else if (decp_data[r] == currentHeight and r<i) {
                            lowerStar.emplace_back(r);
                        }

                        if (decp_data[r] > currentHeight) {
                            upperStar.emplace_back(r);
                        } else if (decp_data[r] == currentHeight and r>i) {
                            upperStar.emplace_back(r);
                        }

                        if (input_data[r] < or_currentHeight) {
                            or_lowerStar.emplace_back(r);
                        } 
                        else if (input_data[r] == or_currentHeight and r<i) {
                            or_lowerStar.emplace_back(r);
                        }

                        if (input_data[r] > or_currentHeight) {
                            or_upperStar.emplace_back(r);
                        } else if (input_data[r] == or_currentHeight and r>i) {
                            or_upperStar.emplace_back(r);
                        }

                    }
                
                if(areVectorsDifferent(lowerStar, or_lowerStar) or areVectorsDifferent(upperStar, or_upperStar)){
                    all_saddle[count_f_saddle] = i;
                    count_f_saddle+=1;
                    
                }
            }
            
            
        }
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

    std::vector<int> findDifferences(const std::vector<int>& vec1, const std::vector<int>& vec2) {
        std::set<int> set1(vec1.begin(), vec1.end());
        std::set<int> set2(vec2.begin(), vec2.end());
        std::vector<int> differences;

        
        for (const auto& elem : set1) {
            if (set2.find(elem) == set2.end()) {
                differences.push_back(elem);
            }
        }

        
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

    void get_false_stars(int index){
        std::vector<int > lowerStar, upperStar;
        double delta;
        double currentHeight = decp_data[index];
        for (int d =0;d<maxNeighbors;d++) {
            
            
            if(adjacency[index*maxNeighbors+d]==-1) continue;
            int r = adjacency[index*maxNeighbors+d];
            
            
                if (decp_data[r] < currentHeight) {
                    lowerStar.emplace_back(r);
                } 
                else if (decp_data[r] == currentHeight and r<index) {
                    lowerStar.emplace_back(r);
                }

                if (decp_data[r] > currentHeight) {
                    upperStar.emplace_back(r);
                } 
                else if (decp_data[r] == currentHeight and r>index) {
                    upperStar.emplace_back(r);
                }

        }

        std::vector<std::vector<int> > lowerWedges = findWedges(lowerStar);
        std::vector<std::vector<int> > upperWedges = findWedges(upperStar);

        

        std::vector<int > o_lowerStar, o_upperStar;
        currentHeight = input_data[index];
        for (int d =0;d<maxNeighbors;d++) {
            
            
            if(adjacency[index*maxNeighbors+d]==-1) continue;
            int r = adjacency[index*maxNeighbors+d];

                if (input_data[r] < currentHeight) {
                    o_lowerStar.emplace_back(r);
                } 
                else if (input_data[r] == currentHeight and r<index) {
                    o_lowerStar.emplace_back(r);
                }

                if (input_data[r] > currentHeight) {
                    o_upperStar.emplace_back(r);
                } 
                else if (input_data[r] == currentHeight and r>index) {
                    o_upperStar.emplace_back(r);
                }

        }
        std::vector<std::vector<int> > o_lowerWedges = findWedges(o_lowerStar);
        std::vector<std::vector<int> > o_upperWedges = findWedges(o_upperStar);
        
        if(lowerWedges.size()!=o_lowerWedges.size() or areVectorsDifferent(lowerStar, o_lowerStar)){
            std::vector<int> diff = findDifferences(lowerStar,o_lowerStar);
        
            if(lowerStar.size()<=o_lowerStar.size() || lowerWedges.size()<=o_lowerWedges.size()){
                // decrease value of the negative lower star.
                if(lowerStar.size()==o_lowerStar.size()){
                    diff = findDifference(o_lowerStar, lowerStar);
                }
                for(int i:diff){
                    delta = (input_data[i]-bound+decp_data[i])/2 - decp_data[i];
                    double oldValue = d_deltaBuffer[i];
                    if (delta > oldValue) {
                        swap(i, delta);
                    }  

                    
                }

                
            }

            else if(lowerStar.size()>o_lowerStar.size()||lowerWedges.size()>o_lowerWedges.size()){
                // decrease value of the saddle point while > the true lower stars.
                
                double maximum = -DBL_MAX;
                int max_id = -1;
                for(int i:o_lowerStar){
                    if(decp_data[i]>maximum) 
                    {
                        maximum = decp_data[i];
                        max_id = i;
                    }

                }

                double minimum = DBL_MAX;
                int min_id = -1;
                for(int i:diff){
                    if(decp_data[i]<minimum) 
                    {
                        minimum = decp_data[i];
                        min_id = i;
                    }

                }

                delta = (input_data[index]-bound+decp_data[index])/2 - decp_data[index];
                double oldValue = d_deltaBuffer[index];
                if (delta > oldValue) {
                    swap(index, delta);
                }  
                
                

                
            }
        
        }
        if(upperWedges.size()!=o_upperWedges.size()  or areVectorsDifferent(upperStar, o_upperStar)){
            std::vector<int> diff = findDifferences(upperStar,o_upperStar);
            // negative lower stars.
            
            

            if(upperStar.size()>o_upperStar.size() || upperWedges.size()>o_upperWedges.size()){

                for(int i:diff){
                    
                    delta = (input_data[i]-bound+decp_data[i])/2 - decp_data[i];
                    
                    if(decp_data[i]>decp_data[index] or (decp_data[i]==decp_data[index] and i>index)){
                        delta = (input_data[i]-bound+decp_data[i])/2 - decp_data[i];
                        double oldValue = d_deltaBuffer[i];
                        if (delta > oldValue) {
                            swap(i, delta);
                        }  
                    }
                    
                    
                }
            }

            else if(upperStar.size()<=o_upperStar.size() || upperWedges.size()<=o_upperWedges.size()){

                if(upperStar.size()==o_upperStar.size()){
                    diff = findDifference(o_upperStar, upperStar);
                }
                double minimum = DBL_MAX;
                int min_id = -1;
                for(int i:diff){
                    if(decp_data[i]<minimum) 
                    {
                        minimum = decp_data[i];
                        min_id = i;
                    }

                }
                
                delta = decp_data[index] - (decp_data[index]+input_data[index]-bound)/2;
                if(decp_data[index]>decp_data[min_id] or (index>min_id and decp_data[index]==decp_data[min_id])){
                delta = (input_data[index]-bound+decp_data[index])/2 - decp_data[index];
                    double oldValue = d_deltaBuffer[index];
                    if (delta > oldValue) {
                        swap(index, delta);
                    }  
                }
                

                
            }
            
            
        }

        return;
        
        
    }

    void apply_delta(){
        #pragma omp parallel for
        for(int tid = 0; tid < num_Elements; tid++)
        {
            if(d_deltaBuffer[tid] != -2000 * bound){
                
                if(abs(d_deltaBuffer[tid]) > 1e-15 && abs(input_data[tid] - decp_data[tid] - d_deltaBuffer[tid])<=bound) decp_data[tid] += d_deltaBuffer[tid];
                else decp_data[tid] = input_data[tid] - bound;
                
            }
        }
    }

    void c_loops(){
        
        
        classifyAllVertices(dec_vertex_type, decp_data, 1);
        get_false_criticle_points();    
        
        while(count_f_max>0 or count_f_min>0 or count_f_saddle>0){
            std::cout<<"c_loops: "<<count_f_max<<", "<<count_f_min<<", "<<count_f_saddle<<std::endl;
            init_delta();

            #pragma omp parallel for
            for(auto i = 0; i < count_f_max; i ++){
                
                int critical_i = all_max[i];
                c_loop(critical_i);

            }                
                
            #pragma omp parallel for
            for(auto i = 0; i < count_f_min; i ++){

                int critical_i = all_min[i];
                c_loop(critical_i, 1);

            }
                
            #pragma omp parallel for
            for(int i =0;i<count_f_saddle;i++){
                int index = all_saddle[i];
                get_false_stars(index);
            }
            
            
            apply_delta();
            classifyAllVertices(dec_vertex_type, decp_data, 1);
            get_false_criticle_points();    

        }
    }

    void get_vertex_traingle(){
        int numFaces = 2 * (width - 1) * (height - 1);
        for(int i = 0; i<numFaces; i++)
        {
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
                        const int *desManifold, const int *ascManifold){
        
        
        
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
                if(uId < vId)
                    continue;

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
                if(upperMem[upperCursor] != desManifold[u]) {
                    upperMem[++upperCursor] = desManifold[u];
                }
            } else if(islarger(vertexId, u, offset)){
                lowerMem2[lowerCursor2++] = u;
                if(lowerMem[lowerCursor] != ascManifold[u]) {
                    lowerMem[++lowerCursor] = ascManifold[u];
                }
            }
        }

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
        
        if(upperCursor > 1 && computeNumberOfLinkComponents(upperMem2.data(), upperCursor2)> 1)
        {
            if(types[0] == -1) types[0] = 2;
            else types[1] = 2;
        }
            
        
        if(lowerCursor > 1 && computeNumberOfLinkComponents(lowerMem2.data(), lowerCursor2) > 1)
        {
            if(types[0] == -1) types[0] = 1;
            else types[1] = 1;
        }
        

        return 3;
    }

    void getSortedIndexes(const double* vec, int* indexes) {
        
        for (int i = 0; i < num_Elements; ++i) {
            indexes[i] = i;
        }

        
        std::sort(indexes, indexes + num_Elements, [&vec](int i, int j) {
            return vec[i] < vec[j] || (vec[i] == vec[j] && i < j); 
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

    void computeCP(std::vector<std::pair<int, int>> &maximaTriplets, std::vector<std::array<int, 46>> &saddleTriplets, 
                    const double *offset, const int *desManifold, const int *ascManifold, std::vector<int> &saddles2, 
                    std::vector<int> &maximum, int &globalMin, int translation){
        const int data_size = width * height * depth;
        int nSaddle2 = 0;
        int nMax = 0;
        maximum.clear();
        saddles2.clear();
        std::vector<int> saddles1;
        int nSaddle1 = 0;
        #pragma omp parallel for reduction(+:nSaddle2, nMax, nSaddle1)
        for(int VertexId = 0; VertexId < data_size; VertexId ++){
            
            int type[2];
            classifyVertex(VertexId, offset, desManifold, ascManifold, type);
            if(type[0] == 2 || type[1] == 2)
            {
                nSaddle2++;
                saddles2.push_back(VertexId);
            }
            
            if(type[0] == 1 || type[1] == 1)
            {
                nSaddle1++;
                saddles1.push_back(VertexId);
            }

            if(type[0] == 4)
            {
                nMax++;
                maximum.push_back(VertexId);
            }
        }
       
        
        
        std::sort(saddles2.begin(), saddles2.end(), [&offset](int v, int u) {
            return islarger(v, u, offset);
        });

        std::sort(maximum.begin(), maximum.end(), [&offset](int v, int u) {
            return isless(v, u, offset);
        });

        

        saddleTriplets.resize(nSaddle2);
        
        
        // int globalMax = sortedVertex[sortedVertex.size()-1];

        int *tempArray = new int[data_size];
        for(int i = 0; i < nMax; i++) {
            tempArray[maximum[i]] = i;
            // std::cout<<input_data[maximum[i]]<<std::endl;
        }
        
        findAscPaths(saddleTriplets, maximum, saddles2, saddles2.data(), maximum.data(), offset,
                    desManifold, tempArray);
        
        if(translation == 0)
        {
            int q = 0;
            for(auto &item:saddleTriplets)
            {
                for(int i = 0; i< item[44]; i++)
                {   
                    item[i] = maximum[item[i]];
                }
                item[45] = saddles2[q++];
            }
            return;
        }

        int *sortedVertex = new int[data_size];
        getSortedIndexes(offset, sortedVertex);
        globalMin = sortedVertex[0];

        return; 
    }

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

        int counter = 0;
        for(auto &item:maximaTriplets)
        {
            if(counter != maximaTriplets.size())
            {
                item.first = saddles2[item.first];
                item.second = Maximum[item.second];
                counter++;
            }
            else
            {
                item.second = Maximum[item.second];
                counter++;
            }
            
        }

        
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

    void get_wrong_neighbors(int saddle, int &num_false_cases, const double *offset)
    {
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
                        
                    }
                }
                label_count++;
            }

        }
    }

    int fixpath(int i, int direction = 0){
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
            
            double d = ((input_data[false_index] - bound) + decp_data[false_index]) / 2.0 - decp_data[false_index];
            
            double oldValue = d_deltaBuffer[false_index];
            if (d > oldValue) {
                swap(false_index, d);
            }  

            return 0;
            
        }

        else 
        {
            
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

            double d = ((input_data[true_index] - bound) + decp_data[true_index]) / 2.0 - decp_data[true_index];
            double oldValue = d_deltaBuffer[true_index];
            if (d > oldValue) {
                swap(true_index, d);
            }  

            return 0;
            
        
        }
        return 0;
    }

    void r_loops(){
        
        init_delta();
        for(int i = 0;i< number_of_false_cases; i++)
        {
            int id = wrong_neighbors_index[i];
            fixpath(id, 0);
        }
        apply_delta();
    }

    void get_wrong_index_max(){
        wrong_max_counter = 0;
        std::fill(wrong_rank_max, wrong_rank_max + num_Elements, 0);
        for(int i = 0; i<dec_saddleTriplets.size(); i++)
        {

            if(saddleTriplets[i][0] != dec_saddleTriplets[i][0])
            {
                int maxId = saddleTriplets[i][0];
                if(wrong_rank_max[maxId] == 0)
                {
                    wrong_rank_max_index[wrong_max_counter * 2] = saddleTriplets[i][0];
                    wrong_rank_max_index[wrong_max_counter * 2 + 1] = dec_saddleTriplets[i][0];
                    wrong_rank_max[maxId] = 1;
                    // std::cout<<saddleTriplets[i][0]<<", "<<dec_saddleTriplets[i][0]<<std::endl;
                    // std::cout<<input_data[saddleTriplets[i][0]]<<", "<<input_data[dec_saddleTriplets[i][0]]<<std::endl;
                    // std::cout<<decp_data[saddleTriplets[i][0]]<<", "<<decp_data[dec_saddleTriplets[i][0]]<<std::endl;
                    wrong_max_counter ++;
                    
                }
            }
            
        }
    }

    void fix_wrong_index_max(int true_index, int false_index){
        double tmp_delta;
        double tmp_true_value = decp_data[true_index];
        double tmp_false_value = decp_data[false_index];
        while(tmp_true_value > tmp_false_value || 
                (tmp_true_value == tmp_false_value && true_index > false_index)){
            tmp_delta = (input_data[true_index] - bound + tmp_true_value) / 2 - tmp_true_value;
            tmp_true_value += tmp_delta;
        }
        
        double d = tmp_true_value - decp_data[true_index];
        if(true_index==4)std::cout<<"delta is:"<<decp_data[true_index]<<", "<<decp_data[false_index]<<", "<<d<<std::endl;
        double oldValue = d_deltaBuffer[true_index];
        
        if (d > oldValue) {
            swap(true_index, d);
        }  

        return;
    }

    void s_loops(){
        get_wrong_index_max();
        init_delta();

        for(int i = 0; i < wrong_max_counter; i++)
        {
            
            int true_index = wrong_rank_max_index[i*2];
            int false_index = wrong_rank_max_index[i*2+1];
            // if(i == 0) {
            //     std::cout<<true_index<< ", "<<false_index<<std::endl;
            //     std::cout<<input_data[true_index]<< ", "<<input_data[false_index]<<std::endl;
            // }
            fix_wrong_index_max(true_index, false_index);
        }

        apply_delta();

    }

    void compute_MergeT(std::vector<Branch> &branches, 
                        const std::vector<int> saddles2, 
                        const std::vector<int> maximum,
                        std::vector<std::pair<int, int>> &maximaTriplets,
                        std::vector<std::array<int, 46>> &saddleTriplets,
                        const double *offset,
                        const int data_size){

        int nMax = maximum.size();
        std::vector<std::pair<int, int>> persistencePairs;
        maximaTriplets.resize(nMax);
        persistencePairs.resize(nMax);
        
        constructPersistencePairs(persistencePairs, maximaTriplets, saddleTriplets, maximum.data(), saddles2.data(), nMax, globalMin);
        branches.resize(nMax);
        std::cout<<nMax<<std::endl;
        int* order = getSortedPositions(offset, data_size);
        
        constructMergeTree(branches, maximaTriplets, maximum,
                           saddles2, order);
    }


    

    

}