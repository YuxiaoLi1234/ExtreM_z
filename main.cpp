#include "ExtreM.h"
#include <ostream>
#include <iostream>
#include <string>
#include <sstream>
#include <parallel/algorithm>  
#include <omp.h>
#include <iomanip>

int main(int argc, char** argv) {
    omp_set_num_threads(128);
    std::cout << std::fixed << std::setprecision(16);
    std::string dimension = argv[1];
    double er = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    std::string file_path;
    
    std::istringstream iss(dimension);
    char delimiter;
    maxNeighbors = 6;
    bool preserve_join_tree = true;

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
    decp_data_copy = new double[num_Elements]; 

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
    delta_counter.resize(num_Elements, 0);
    lowerStars = new int[(maxNeighbors+1) * num_Elements];
    upperStars = new int[(maxNeighbors+1) * num_Elements];

    dec_lowerStars = new int[(maxNeighbors+1) * num_Elements];
    dec_upperStars = new int[(maxNeighbors+1) * num_Elements];
    const std::string tree_type = "split";

    getdata(file_path, er, num_Elements, tree_type);

    int globalMax;
    int *sortedVertex1 = new int[num_Elements];
    getSortedIndexes(input_data, sortedVertex1);
    globalMin = sortedVertex1[0];
    globalMax = sortedVertex1[num_Elements-1];
    std::cout<<globalMin<<","<<globalMax<<std::endl;

    decp_data[globalMin] = input_data[globalMin] - bound;
    decp_data[globalMax] = input_data[globalMax] - bound;

    adjacency = new int[num_Elements * maxNeighbors];
    
    computeAdjacency();
    get_vertex_traingle();

    int *DS_M = new int[num_Elements];
    int *AS_M = new int[num_Elements];

    classifyAllVertices(vertex_type, input_data, lowerStars, upperStars, 0);
    classifyAllVertices(dec_vertex_type, decp_data, dec_lowerStars, dec_upperStars, 1);
    
    ComputeDescendingManifold(input_data, vertex_type, DS_M);
    ComputeAscendingManifold(input_data, vertex_type, AS_M);

    
    std::cout<<"conpuattion of manifold over"<<std::endl;

    std::vector<std::pair<int, int>> maximaTriplets, minimumTriplets;
    std::vector<std::pair<int, int>> dec_maximaTriplets, dec_minimumTriplets;
    
    
    
    std::vector<Branch> branches, dec_branches;
    computeCP(maximaTriplets, minimumTriplets, saddleTriplets, saddle1Triplets, input_data, DS_M, AS_M, vertex_type, 
    saddles2, saddles1, minimum, maximum, globalMin, 0);
    
    int nMax = maximum.size();
    int nMin = minimum.size();

    // computeCP(maximaTriplets, saddleTriplets, input_data, DS_M, vertex_type, saddles2, maximum, globalMin, 0);
    // compute_MergeT(branches, saddles2, maximum, maximaTriplets, saddleTriplets, decp_data, num_Elements, globalMin);

    

    computelargestSaddlesForMax(nMax, saddleTriplets, largestSaddlesForMax, &maximum);
    computesmallestSaddlesForMin(nMin, saddle1Triplets, smallestSaddlesForMin, &minimum);

    std::sort(saddleTriplets.begin(), saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
        return a[45] < b[45];  // 按最后一个元素升序排序
    });
    std::sort(saddle1Triplets.begin(), saddle1Triplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
        return a[45] < b[45];  // 按最后一个元素升序排序
    });

    
    for(int i = 0; i<saddles2.size(); i++){
        decp_data[saddles2[i]] = input_data[saddles2[i]] - bound;
        // decp_data[saddles1[i]] = input_data[saddles1[i]] - bound;
    }

    if(preserve_join_tree){
        for(int i = 0; i<saddles1.size(); i++){
            decp_data[saddles1[i]] = input_data[saddles1[i]] - bound;
        }
    }
    
    
    
    std::cout<<"preserving cp started! "<<std::endl;
    
    std::cout<<"preserving cp started! "<<std::endl;
    // preserving the critical points, including saddles
    c_loops(tree_type);
    std::cout<<"preserving cp ended!"<<std::endl;

    or_saddle_max_map = new int[num_Elements * 4];
    wrong_neighbors = new int[num_Elements]();
    wrong_neighbors_index = new int[num_Elements]();

    wrong_rank_max = new int[num_Elements]();
    wrong_rank_max_index = new int[num_Elements * 2]();

    wrong_rank_max_2 = new int[num_Elements]();
    wrong_rank_max_index_2 = new int[num_Elements * 2]();

    wrong_rank_saddle = new int[num_Elements]();
    wrong_rank_saddle_index = new int[num_Elements * 2]();

    or_saddle_min_map = new int[num_Elements * 4];
    wrong_neighbors_ds = new int[num_Elements]();
    wrong_neighbors_ds_index = new int[num_Elements]();

    wrong_rank_min = new int[num_Elements]();
    wrong_rank_min_index = new int[num_Elements * 2]();

    wrong_rank_min_2 = new int[num_Elements]();
    wrong_rank_min_index_2 = new int[num_Elements * 2]();

    wrong_rank_saddle_join = new int[num_Elements]();
    wrong_rank_saddle_join_index = new int[num_Elements * 2]();


    int *dec_DS_M = new int[num_Elements];
    int *dec_AS_M = new int[num_Elements];

    ComputeDescendingManifold(decp_data, dec_vertex_type, dec_DS_M);
    ComputeAscendingManifold(decp_data, dec_vertex_type, dec_AS_M);
    std::cout<<"compuattion of manifold over"<<std::endl;


    
    computeCP(dec_maximaTriplets, dec_minimumTriplets, dec_saddleTriplets, dec_saddle1Triplets, decp_data, dec_DS_M, dec_AS_M, dec_vertex_type, 
    dec_saddles2, dec_saddles1, dec_minimum, dec_maximum, dec_globalMin, 0);
    // computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_vertex_type, dec_saddles2, dec_maximum, dec_globalMin, 0);

    std::cout<<"cp extraction ended"<<std::endl;

    std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
        return a[45] < b[45];  // 按最后一个元素升序排序
    });
    

    std::sort(dec_saddle1Triplets.begin(), dec_saddle1Triplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
        return a[45] < b[45];  // 按最后一个元素升序排序
    });

    
    // compute the saddle labels for the original data;
    # pragma omp parallel for 
    for(int i = 0;i < saddleTriplets.size(); i++)
    {
        int saddle = saddleTriplets[i][45];
        compute_Max_for_Saddle(saddle, input_data);
    }

    # pragma omp parallel for 
    for(int i = 0;i < saddle1Triplets.size(); i++)
    {
        int saddle = saddle1Triplets[i][45];
        compute_Min_for_Saddle(saddle, input_data);
    }

    std::cout<<"ranking ended"<<std::endl;
    number_of_false_cases = 0;
    number_of_false_cases1 = 0;
    std::fill(wrong_neighbors, wrong_neighbors + num_Elements, 0);
    std::fill(wrong_neighbors_ds, wrong_neighbors_ds + num_Elements, 0);

    for(int i = 0;i < saddleTriplets.size(); i++)
    {
        int saddle = saddleTriplets[i][45];
        get_wrong_split_neighbors(saddle, number_of_false_cases, decp_data );
    }

    if(preserve_join_tree){
        for(int i = 0;i < saddle1Triplets.size(); i++){
            int saddle = saddle1Triplets[i][45];
            get_wrong_join_neighbors(saddle, number_of_false_cases1, decp_data);
        }
    }
    

    std::cout<<number_of_false_cases<<std::endl;
    if(number_of_false_cases == 0){
        
        ComputeDescendingManifold(decp_data, dec_vertex_type, dec_DS_M);
        computeCP(dec_maximaTriplets, dec_minimumTriplets, dec_saddleTriplets, dec_saddle1Triplets, decp_data, dec_DS_M, dec_AS_M, dec_vertex_type, 
        dec_saddles2, dec_saddles1, dec_minimum, dec_maximum, dec_globalMin, 0);
        // computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_vertex_type, dec_saddles2, dec_maximum, dec_globalMin, 0);

        computelargestSaddlesForMax(nMax, dec_saddleTriplets, dec_largestSaddlesForMax, &dec_maximum);
        // computesmallestSaddlesForMin(nMin, dec_saddle1Triplets, dec_smallestSaddlesForMin, &dec_minimum);

        std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a[45] < b[45];  
        });
        // std::sort(dec_saddle1Triplets.begin(), dec_saddle1Triplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
        //     return a[45] < b[45];  // 按最后一个元素升序排序
        // });

        
        s_loops(tree_type);

        if(wrong_max_counter == 0){
            saddle_loops(tree_type);
        }
        
    }
    if(number_of_false_cases1 == 0 && preserve_join_tree){
        
        ComputeDescendingManifold(decp_data, dec_vertex_type, dec_DS_M);
        ComputeAscendingManifold(decp_data, dec_vertex_type, dec_AS_M);
        computeCP(dec_maximaTriplets, dec_minimumTriplets, dec_saddleTriplets, dec_saddle1Triplets, decp_data, dec_DS_M, dec_AS_M, dec_vertex_type, 
        dec_saddles2, dec_saddles1, dec_minimum, dec_maximum, dec_globalMin, 0);

        computesmallestSaddlesForMin(nMin, dec_saddle1Triplets, dec_smallestSaddlesForMin, &dec_minimum);

        std::sort(dec_saddle1Triplets.begin(), dec_saddle1Triplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a[45] < b[45];  
        });

        
        s_loops_join(tree_type);
        if(wrong_min_counter == 0){
            saddle_loops_join(tree_type);
        }
        
    }
    // get_false_criticle_points();
    
    while(number_of_false_cases > 0 || number_of_false_cases1 > 0|| count_f_max > 0 || count_f_min > 0 || count_f_saddle > 0 
            || wrong_max_counter > 0 || wrong_max_counter_2 > 0 || wrong_saddle_counter > 0
            || wrong_min_counter > 0 || wrong_min_counter_2 > 0 || wrong_saddle_counter_join > 0)
    {
       
        std::cout<<
        "whole loops:"<<number_of_false_cases<<", "<<number_of_false_cases1<<", "<<count_f_max <<", "<< count_f_min << ", "<<count_f_saddle<<", "
        <<wrong_max_counter<<", "<<wrong_max_counter_2<<", "<<wrong_saddle_counter
        <<", "<<wrong_min_counter<<", "<<wrong_min_counter_2<<", "<<wrong_saddle_counter_join<<std::endl;
        r_loops(tree_type);
        c_loops(tree_type);

        
        ComputeDescendingManifold(decp_data, dec_vertex_type, dec_DS_M);
        ComputeAscendingManifold(decp_data, dec_vertex_type, dec_AS_M);
        computeCP(dec_maximaTriplets, dec_minimumTriplets, dec_saddleTriplets, dec_saddle1Triplets, decp_data, dec_DS_M, dec_AS_M, dec_vertex_type, 
        dec_saddles2, dec_saddles1, dec_minimum, dec_maximum, dec_globalMin, 0);

        // computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_vertex_type, dec_saddles2, dec_maximum, dec_globalMin, 0);
        computelargestSaddlesForMax(nMax, dec_saddleTriplets, dec_largestSaddlesForMax, &dec_maximum);
        computesmallestSaddlesForMin(nMin, dec_saddle1Triplets, dec_smallestSaddlesForMin, &dec_minimum);

        std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a[45] < b[45];  // 按最后一个元素升序排序
        });
        std::sort(dec_saddle1Triplets.begin(), dec_saddle1Triplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a[45] < b[45];  // 按最后一个元素升序排序
        });
        
        
        // compute the saddle labels for the decp_data;

        number_of_false_cases = 0;
        number_of_false_cases1 = 0;
        std::fill(wrong_neighbors, wrong_neighbors + num_Elements, 0);
        std::fill(wrong_neighbors_ds, wrong_neighbors_ds + num_Elements, 0);
        for(int i = 0;i < saddleTriplets.size(); i++){
            int saddle = saddleTriplets[i][45];
            get_wrong_split_neighbors(saddle, number_of_false_cases, decp_data );
        }
        
        if(preserve_join_tree){
            for(int i = 0;i < saddle1Triplets.size(); i++){
                int saddle = saddle1Triplets[i][45];
                get_wrong_join_neighbors(saddle, number_of_false_cases1, decp_data );
            }
        }
        
        
        if(number_of_false_cases == 0){
            s_loops(tree_type);
            if(wrong_max_counter == 0){
                saddle_loops(tree_type);
            }
        }

        if(number_of_false_cases1 == 0 && preserve_join_tree){
            s_loops_join(tree_type);
            if(wrong_min_counter == 0){
                saddle_loops_join(tree_type);
            }
        }
        
        c_loops(tree_type);

        ComputeDescendingManifold(decp_data, dec_vertex_type, dec_DS_M);
        ComputeAscendingManifold(decp_data, dec_vertex_type, dec_AS_M);
        computeCP(dec_maximaTriplets, dec_minimumTriplets, dec_saddleTriplets, dec_saddle1Triplets, decp_data, dec_DS_M, dec_AS_M, dec_vertex_type, 
        dec_saddles2, dec_saddles1, dec_minimum, dec_maximum, dec_globalMin, 0);
        // computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_vertex_type, dec_saddles2, dec_maximum, dec_globalMin, 0);
        computelargestSaddlesForMax(nMax, dec_saddleTriplets, dec_largestSaddlesForMax, &dec_maximum);
        computesmallestSaddlesForMin(nMin, dec_saddle1Triplets, dec_smallestSaddlesForMin, &dec_minimum);

        std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a[45] < b[45];  // 按最后一个元素升序排序
        });
        std::sort(dec_saddle1Triplets.begin(), dec_saddle1Triplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a[45] < b[45];  // 按最后一个元素升序排序
        });

        // compute the saddle labels for the decp_data;

        number_of_false_cases = 0;
        number_of_false_cases1 = 0;
        std::fill(wrong_neighbors, wrong_neighbors + num_Elements, 0);
        std::fill(wrong_neighbors_ds, wrong_neighbors_ds + num_Elements, 0);
        for(int i = 0;i < saddleTriplets.size(); i++){
            int saddle = saddleTriplets[i][45];
            get_wrong_split_neighbors(saddle, number_of_false_cases, decp_data );
        }


        if(preserve_join_tree){
            for(int i = 0;i < saddle1Triplets.size(); i++){
                int saddle = saddle1Triplets[i][45];
                get_wrong_join_neighbors(saddle, number_of_false_cases1, decp_data );
            }
        }

        if(number_of_false_cases == 0){
            s_loops(tree_type);
            if(wrong_max_counter == 0){
                saddle_loops(tree_type);
            }
        }

        if(number_of_false_cases1 == 0 && preserve_join_tree){
            
            s_loops_join(tree_type);
            if(wrong_min_counter == 0){
                saddle_loops_join(tree_type);
            }
        }

        c_loops(tree_type);

        ComputeDescendingManifold(decp_data, dec_vertex_type, dec_DS_M);
        ComputeAscendingManifold(decp_data, dec_vertex_type, dec_AS_M);
        computeCP(dec_maximaTriplets, dec_minimumTriplets, dec_saddleTriplets, dec_saddle1Triplets, decp_data, dec_DS_M, dec_AS_M, dec_vertex_type, 
        dec_saddles2, dec_saddles1, dec_minimum, dec_maximum, dec_globalMin, 0);

        // computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_vertex_type, dec_saddles2, dec_maximum, dec_globalMin, 0);

        computelargestSaddlesForMax(nMax, dec_saddleTriplets, dec_largestSaddlesForMax, &dec_maximum);
        computesmallestSaddlesForMin(nMin, dec_saddle1Triplets, dec_smallestSaddlesForMin, &dec_minimum);

        std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a[45] < b[45];  // 按最后一个元素升序排序
        });
        std::sort(dec_saddle1Triplets.begin(), dec_saddle1Triplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
            return a[45] < b[45];  // 按最后一个元素升序排序
        });

        // compute the saddle labels for the decp_data;




        number_of_false_cases = 0;
        number_of_false_cases1 = 0;
        std::fill(wrong_neighbors, wrong_neighbors + num_Elements, 0);
        std::fill(wrong_neighbors_ds, wrong_neighbors_ds + num_Elements, 0);
        for(int i = 0;i < saddleTriplets.size(); i++)
        {
            int saddle = saddleTriplets[i][45];
            get_wrong_split_neighbors(saddle, number_of_false_cases, decp_data );
        }

        if(preserve_join_tree){
            for(int i = 0;i < saddle1Triplets.size(); i++){
                int saddle = saddle1Triplets[i][45];
                get_wrong_join_neighbors(saddle, number_of_false_cases1, decp_data );
            }
        }
        
        if(number_of_false_cases == 0){
            
            get_wrong_index_max();
            get_wrong_index_saddles();

        }

        if(number_of_false_cases1 == 0 && preserve_join_tree){
            
            get_wrong_index_min();
            get_wrong_index_saddles_join();

        }
        
    }
    

    // std::sort(dec_saddleTriplets.begin(), dec_saddleTriplets.end(), [](const std::array<int, 46>& a, const std::array<int, 46>& b) {
    //     return a[45] < b[45];  // 按最后一个元素升序排序
    // });
    
    std::cout<<"ended"<<std::endl;
    computeCP(dec_maximaTriplets, dec_minimumTriplets, dec_saddleTriplets, dec_saddle1Triplets, decp_data, dec_DS_M, dec_AS_M, dec_vertex_type, 
    dec_saddles2, dec_saddles1, dec_minimum, dec_maximum, dec_globalMin, 1);
    // computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_vertex_type, dec_saddles2, dec_maximum, dec_globalMin, 0);
    compute_MergeT(dec_branches, dec_saddles2, dec_maximum, dec_maximaTriplets, dec_saddleTriplets, decp_data, num_Elements, dec_globalMin);
        
    // computeCP(dec_maximaTriplets, dec_saddleTriplets, decp_data, dec_DS_M, dec_vertex_type, dec_saddles2, dec_maximum, dec_globalMin, 1);
    // compute_MergeT(dec_branches, dec_saddles2, dec_maximum, dec_maximaTriplets, dec_saddleTriplets, decp_data, num_Elements, dec_globalMin);
        
    
    // computeCP(maximaTriplets, saddleTriplets, input_data, DS_M, vertex_type, saddles2, maximum, globalMin, 1);
    // compute_MergeT(branches, saddles2, maximum, maximaTriplets, saddleTriplets, input_data, num_Elements, globalMin);
    
    computeCP(maximaTriplets, minimumTriplets, saddleTriplets, saddle1Triplets, decp_data, DS_M, AS_M, vertex_type, 
    saddles2, saddles1, minimum, maximum, globalMin, 1);
    // computeCP(maximaTriplets, saddleTriplets, input_data, DS_M, AS_M, saddles2, maximum, globalMin, 0);
    compute_MergeT(branches, saddles2, maximum, maximaTriplets, saddleTriplets, input_data, num_Elements, globalMin);
    
    
    
    double cnt = 0.0;
    for(int i = 0; i<num_Elements; i++)
    {
        if(decp_data[i] != decp_data_copy[i]) cnt++;
    }
    std::cout<<cnt/num_Elements<<std::endl;

    int branchNumber = branches.size();
    
    int pointIds[2];
    int dec_pointIds[2];
    for(int i = 0; i<branchNumber; i++){
        auto branch = branches[i];
        auto dec_branch = dec_branches[i];
        auto &vertices = branch.vertices;
        auto &dec_vertices = dec_branch.vertices;
        const int verticesNumber = vertices.size();
        
        for(int p = 0; p < verticesNumber - 1; p++){
            pointIds[0] = vertices[p].second;
            pointIds[1] = vertices[p + 1].second;

            dec_pointIds[0] = dec_vertices[p].second;
            dec_pointIds[1] = dec_vertices[p + 1].second;

            if(pointIds[0] != dec_pointIds[0] || pointIds[1] != dec_pointIds[1]){
                std::cout<<"branch "<<i<<" unmatched at vertex: "<<p<<std::endl;
                std::cout<<"original is: "<<pointIds[0]<<", "<<pointIds[1]<<std::endl;
                std::cout<<"fixed decp is: "<<dec_pointIds[0]<<", "<<dec_pointIds[1]<<std::endl;
            }
        }
    }

    
    saveArrayToBin(decp_data, num_Elements, "fixed.bin");
    delete[] input_data;
    delete[] decp_data;
    delete[] decp_data_copy;
    


    return 0;
}
