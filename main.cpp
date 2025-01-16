#include "ExtreM.h"
#include <ostream>
#include <iostream>
#include <string>
#include <sstream>
#include <parallel/algorithm>  
#include <omp.h>

int main(int argc, char** argv) {
    omp_set_num_threads(44);
    // std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::string dimension = argv[1];
    double er = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    std::string file_path;
    
    std::istringstream iss(dimension);
    char delimiter;
    maxNeighbors = 6;
    
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
    lowerStars = new int[(maxNeighbors+1) * num_Elements];
    upperStars = new int[(maxNeighbors+1) * num_Elements];

    dec_lowerStars = new int[(maxNeighbors+1) * num_Elements];
    dec_upperStars = new int[(maxNeighbors+1) * num_Elements];
    
    getdata(file_path, er, num_Elements);

    
    
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
    
    // std::vector<int> saddles2, dec_saddles2;

    std::vector<Branch> branches, dec_branches;
    computeCP(maximaTriplets, saddleTriplets, input_data, DS_M, AS_M, saddles2, maximum, globalMin, 0);
    // compute_MergeT(branches, saddles2, maximum, maximaTriplets, saddleTriplets, decp_data, num_Elements, globalMin);

    std::sort(saddleTriplets.begin(), saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
        return a.back() < b.back();  // 按最后一个元素升序排序
    });

    std::cout<<"preserving cp started! "<<std::endl;
    classifyAllVertices(vertex_type, input_data, lowerStars, upperStars, 0);
    classifyAllVertices(dec_vertex_type, decp_data, dec_lowerStars, dec_upperStars, 1);
    std::cout<<"preserving cp started! "<<std::endl;
    // preserving the critical points, including saddles
    c_loops();
    std::cout<<"preserving cp ended!"<<std::endl;

    or_saddle_max_map = new int[num_Elements * 4];
    wrong_neighbors = new int[num_Elements]();
    wrong_neighbors_index = new int[num_Elements]();
    wrong_rank_max = new int[num_Elements]();
    wrong_rank_max_index = new int[num_Elements * 2]();

    int *dec_DS_M = new int[num_Elements];
    int *dec_AS_M = new int[num_Elements];

    ComputeDescendingManifold(decp_data, dec_DS_M);
    ComputeAscendingManifold(decp_data, dec_AS_M);
    std::cout<<"compuattion of manifold over"<<std::endl;

    int *dec_cpMap = new int[num_Elements];
    
    computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_AS_M, dec_saddles2, dec_maximum, dec_globalMin, 0);
    std::cout<<"cp extraction ended"<<std::endl;



    

    std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
        return a.back() < b.back();  // 按最后一个元素升序排序
    });

    // compute the saddle labels for the original data;
    # pragma omp parallel for 
    for(int i = 0;i < saddleTriplets.size(); i++)
    {
        int saddle = saddleTriplets[i][45];
        compute_Max_for_Saddle(saddle, input_data );
    }


    number_of_false_cases = 0;
    std::fill(wrong_neighbors, wrong_neighbors + num_Elements, 0);
    for(int i = 0;i < saddleTriplets.size(); i++)
    {
        int saddle = saddleTriplets[i][45];
        get_wrong_neighbors(saddle, number_of_false_cases, decp_data );
    }

    std::cout<<number_of_false_cases<<std::endl;

    // get_false_criticle_points();
    
    while(number_of_false_cases > 0 || count_f_max > 0 || count_f_min > 0 || count_f_saddle > 0 || wrong_max_counter > 0)
    {
       
        std::cout<<"whole loops:"<<number_of_false_cases<<", "<<count_f_max <<", "<<count_f_min << ", "<<count_f_saddle<<", "<<wrong_max_counter<<std::endl;
        r_loops();
        c_loops();
        

        ComputeDescendingManifold(decp_data, dec_DS_M);
        ComputeAscendingManifold(decp_data, dec_AS_M);
        computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_AS_M, dec_saddles2, dec_maximum, dec_globalMin, 0);
        std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a.back() < b.back();  // 按最后一个元素升序排序
        });

        // compute the saddle labels for the decp_data;

        number_of_false_cases = 0;
        std::fill(wrong_neighbors, wrong_neighbors + num_Elements, 0);
        for(int i = 0;i < saddleTriplets.size(); i++)
        {
            int saddle = saddleTriplets[i][45];
            get_wrong_neighbors(saddle, number_of_false_cases, decp_data);
        }
        
        if(number_of_false_cases == 0)
        {
            ComputeDescendingManifold(decp_data, dec_DS_M);
            ComputeAscendingManifold(decp_data, dec_AS_M);
            computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_AS_M, dec_saddles2, dec_maximum, dec_globalMin, 0);
            std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
                return a.back() < b.back();  // 按最后一个元素升序排序
            });
            s_loops();
            std::cout<<"s_loops completed"<<std::endl;
        }
        

        c_loops();

        ComputeDescendingManifold(decp_data, dec_DS_M);
        ComputeAscendingManifold(decp_data, dec_AS_M);
        computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_AS_M, dec_saddles2, dec_maximum, dec_globalMin, 0);
        std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a.back() < b.back();  // 按最后一个元素升序排序
        });

        // compute the saddle labels for the decp_data;

        number_of_false_cases = 0;
        std::fill(wrong_neighbors, wrong_neighbors + num_Elements, 0);
        for(int i = 0;i < saddleTriplets.size(); i++)
        {
            int saddle = saddleTriplets[i][45];
            get_wrong_neighbors(saddle, number_of_false_cases, decp_data);
        }

        if(number_of_false_cases == 0)
        {
            ComputeDescendingManifold(decp_data, dec_DS_M);
            ComputeAscendingManifold(decp_data, dec_AS_M);
            computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_AS_M, dec_saddles2, dec_maximum, dec_globalMin);
            std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
                return a.back() < b.back();  // 按最后一个元素升序排序
            });
            s_loops();
        }

        c_loops();

        ComputeDescendingManifold(decp_data, dec_DS_M);
        ComputeAscendingManifold(decp_data, dec_AS_M);
        computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_AS_M, dec_saddles2, dec_maximum, dec_globalMin);
        std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a.back() < b.back();  // 按最后一个元素升序排序
        });

        // compute the saddle labels for the decp_data;

        number_of_false_cases = 0;
        std::fill(wrong_neighbors, wrong_neighbors + num_Elements, 0);
        for(int i = 0;i < saddleTriplets.size(); i++)
        {
            int saddle = saddleTriplets[i][45];
            get_wrong_neighbors(saddle, number_of_false_cases, decp_data);
        }
        
        if(number_of_false_cases == 0)
        {
            ComputeDescendingManifold(decp_data, dec_DS_M);
            ComputeAscendingManifold(decp_data, dec_AS_M);
            computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_AS_M, dec_saddles2, dec_maximum, dec_globalMin);
            std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
                return a.back() < b.back();  // 按最后一个元素升序排序
            });
            get_wrong_index_max();
        }
        
    }


    std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
        return a.back() < b.back();  // 按最后一个元素升序排序
    });
    for(int i = 0; i<saddleTriplets.size();i++)
    {
        if(saddleTriplets[i][0] != dec_saddleTriplets[i][0]) std::cout<<i<<std::endl;
    }

    
    computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_AS_M, dec_saddles2, dec_maximum, dec_globalMin, 1);
    compute_MergeT(dec_branches, dec_saddles2, dec_maximum, dec_maximaTriplets, dec_saddleTriplets, decp_data, num_Elements, dec_globalMin);

    delete[] input_data;
    delete[] decp_data;

    return 0;
}
