#include "ExtreM.h"
#include <ostream>
#include <iostream>
#include <string>
#include <sstream>
#include <parallel/algorithm>  
#include <omp.h>

int main(int argc, char** argv) {
    omp_set_num_threads(128);
    // std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::string dimension = argv[1];
    double er = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    std::string file_path;
    
    std::istringstream iss(dimension);
    char delimiter;
    maxNeighbors = 12;
    
    if (std::getline(iss, file_path, ',')) {
        
        if (iss >> width >> delimiter && delimiter == ',' &&
            iss >> height >> delimiter && delimiter == ',' &&
            iss >> depth) {
            std::cout << "Filepath: " << file_path << std::endl;
            std::cout << "Width: " << width << std::endl;
            std::cout << "Height: " << height << std::endl;
            std::cout << "Depth: " << depth << std::endl;
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for file" << std::endl;
    }

    
    // std::string filename = extractFilename(file_path);
    // extern double* d_deltaBuffer;
    // extern int width, height, depth, maxNeighbors, num_Elements, count_f_max, count_f_min, count_f_saddle;
    // extern int* adjacency, *dec_vertex_type, *vertex_type, *all_max, *all_min, *all_saddle;
    // extern double *decp_data, *input_data;
    // extern double bound;
    // extern std::vector<std::vector<int>> vertex_cells;
    num_Elements = width*height*depth;
    std::cout<<"size is:"<<num_Elements<<std::endl;
    input_data = new double[num_Elements];      
    decp_data = new double[num_Elements];  

    d_deltaBuffer = new double[num_Elements];
    count_f_max = 0;
    count_f_min = 0;
    count_f_saddle = 0;
    dec_vertex_type = new int[num_Elements];
    vertex_type = new int[num_Elements];
    all_max = new int[num_Elements];
    all_min = new int[num_Elements];
    all_saddle = new int[num_Elements];
    vertex_cells.resize(2 * (width - 1) * (height - 1));


    
    getdata(file_path, er, bound, num_Elements);

    
    
    std::cout << "Original Data: " << input_data[0] << ", " << input_data[1] << ", " << input_data[2] << std::endl;
    std::cout << "Decompressed Data: " << decp_data[0] << ", " << decp_data[1] << ", " << decp_data[2] << std::endl;
    
    // int *adjacency = new int[num_Elements * maxNeighbors];
    adjacency = new int[num_Elements * maxNeighbors];
    
    computeAdjacency();
    get_vertex_traingle();

    int *DS_M = new int[num_Elements];
    int *AS_M = new int[num_Elements];

    // compute AS/DS manifold
    ComputeDescendingManifold(input_data, DS_M);
    ComputeAscendingManifold(input_data, AS_M);
    
    std::cout<<"conpuattion of manifold over"<<std::endl;
    int *cpMap = new int[num_Elements];
    std::vector<std::pair<int, int>> maximaTriplets;
    std::vector<std::pair<int, int>> dec_maximaTriplets;
    computeCP(maximaTriplets, input_data, DS_M, AS_M, cpMap);

    

    
    vertex_type = new int[num_Elements];
    dec_vertex_type = new int[num_Elements];
    

    std::cout<<"preserving cp started! "<<std::endl;
    classifyAllVertices(vertex_type, input_data, 0);
    std::cout<<"preserving cp started! "<<std::endl;
    // preserving the critical points, including saddles
    c_loops();
    std::cout<<"preserving cp ended!"<<std::endl;

    int *dec_DS_M = new int[num_Elements];
    int *dec_AS_M = new int[num_Elements];

    ComputeDescendingManifold(decp_data, dec_DS_M);
    ComputeAscendingManifold(decp_data, dec_AS_M);
    std::cout<<"compuattion of manifold over"<<std::endl;

    int *dec_cpMap = new int[num_Elements];
    computeCP(dec_maximaTriplets, decp_data, dec_DS_M, dec_AS_M, dec_cpMap);
    std::cout<<"cp extraction ended"<<std::endl;

    std::cout<<"original: "<<std::endl;
    for(auto item:maximaTriplets)
    {
        std::cout<<item.first<<", "<<item.second<<std::endl;
        
    }

    std::cout<<"decp: "<<std::endl;
    for(auto item:dec_maximaTriplets)
    {
        std::cout<<item.first<<", "<<item.second<<std::endl;
        
    }
    
    

    delete[] input_data;
    delete[] decp_data;

    return 0;
}
