#ifndef EXTREM_H
#define EXTREM_H
#include <cstdint>
#include <stddef.h>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <atomic>
struct Branch {
    std::vector<std::pair<int, int>> vertices; // order, globalId, first pair is the maximum
    Branch *parentBranch = nullptr;

    void print() const;
};

extern "C"{
    extern int directions[78], direction_to_index_mapping[26][3]; 
    extern double* d_deltaBuffer;
    extern int width, height, depth, maxNeighbors, num_Elements, number_of_false_cases;
    extern std::atomic_int count_f_max, count_f_min, count_f_saddle;
    extern int wrong_max_counter, globalMin, dec_globalMin, wrong_max_counter_2, wrong_saddle_counter;
    extern int wrong_saddle_counter_join;

    extern int* adjacency, *dec_vertex_type, *vertex_type, *all_max, *all_min, *all_saddle;
    extern double *decp_data, *input_data, *decp_data_copy;
    extern double bound;
    extern int *or_saddle_max_map, *wrong_neighbors, *wrong_neighbors1, *wrong_neighbors_index, *wrong_rank_max, *wrong_rank_max_index, *wrong_rank_saddle, *wrong_rank_saddle_index, *wrong_rank_max_2, *wrong_rank_max_index_2;
    extern int *wrong_rank_ds_saddle, *wrong_rank_ds_saddle_index;
    extern int *or_saddle_min_map, *wrong_neighbors_ds, *wrong_neighbors_ds_index, *wrong_rank_min, *wrong_rank_min_index, *wrong_rank_min_2, *wrong_rank_min_index_2;
    extern int number_of_false_cases1, wrong_min_counter, wrong_min_counter_2;
    extern std::vector<std::array<int, 46>> saddleTriplets, dec_saddleTriplets;
    extern std::vector<std::array<int, 46>> saddle1Triplets, dec_saddle1Triplets;
    extern std::vector<std::vector<int>> vertex_cells;
    extern std::vector<int> saddles2, maximum, dec_saddles2, dec_maximum, delta_counter;
    extern std::vector<int> saddles1, minimum, dec_saddles1, dec_minimum;
    extern std::unordered_map<int, int> largestSaddlesForMax, dec_largestSaddlesForMax;
    extern std::unordered_map<int, int> smallestSaddlesForMin, dec_smallestSaddlesForMin;
    extern int *lowerStars, *upperStars, *dec_lowerStars, *dec_upperStars;

    extern int *wrong_rank_saddle_join, *wrong_rank_saddle_join_index;

    void saveArrayToBin(const double* arr, size_t size, const std::string& filename);
    void get_vertex_traingle();
    void getdata(const std::string &filename, 
                const double er, 
                int data_size, std::string type);

    int ComputeDescendingManifold(const double *offset, const int *vertex_type, int *DS_M);
    int ComputeAscendingManifold(const double *offset, const int *vertex_type, int *AS_M);
    void classifyAllVertices(int *vertex_type, const double* heightMap, int *lowerStars, int *upperStars, int type);
    void computeAdjacency();
    void c_loops(std::string type);
    int classifyVertex(const int vertexId, const double *offset, 
                                        const int *desManifold, const int *ascManifold,
                                        int *types);

    int computeNumberOfLinkComponents(const int* linkVertices, const int nLinkVertices);
    void sortAndRemoveDuplicates(std::array<int, 46> &triplet);
    void computeCP(std::vector<std::pair<int, int>> &maximaTriplets, std::vector<std::pair<int, int>> &minimumTriplets, 
                    std::vector<std::array<int, 46>> &saddleTriplets, std::vector<std::array<int, 46>> &saddle1Triplets, 
                    const double *offset, const int *desManifold, const int *ascManifold,
                    const int *vertex_type_tmp, std::vector<int> &saddles2, 
                    std::vector<int> &saddles1, std::vector<int> &minimum,
                    std::vector<int> &maximum, int &globalMin, int translation = 0);
    // void computeCP(std::vector<std::pair<int, int>> &maximaTriplets, std::vector<std::array<int, 46>> &saddleTriplets, 
    //                 const double *offset, const int *desManifold, const int *vertex_type, std::vector<int> &saddles2, 
    //                 std::vector<int> &maximum, int &globalMin, int translation);

    void getSortedIndexes(const double* vec, int* indexes);
    void findAscPaths(  std::vector<std::array<int, 46>> &saddleTriplets,
                        std::vector<int> &maximaLocalToGlobal,
                        std::vector<int> &saddlesLocalToGlobal,
                        const int *saddles2, const int *maximum,
                        const double *offset,
                        const int *desManifold, const int *ascManifold, int &counter_tmp);

    int constructPersistencePairs(
                        std::vector<std::pair<int, int>> &pairs,
                        std::vector<std::pair<int, int>> &maximaTriplets,
                        std::vector<std::array<int, 46>> &saddleTriplets,
                        const int* maximum,
                        const int* saddles2,
                        const int nMax,
                        int globalMin);
    void compute_Max_for_Saddle(int saddle, const double *offset);
    void compute_Min_for_Saddle(int saddle, const double *offset);
    void get_wrong_split_neighbors(int saddle, int &num_false_cases, const double *offset);
    void get_wrong_join_neighbors(int saddle, int &num_false_cases, const double *offset);
    int fixpath(int i, int direction, std::string type);
    void get_false_criticle_points();
    void r_loops(std::string type);
    void s_loops(std::string type);
    void s_loops_join(std::string type);

    void get_wrong_index_max();
    void compute_MergeT(std::vector<Branch> &branches, 
                        const std::vector<int> saddles2, 
                        const std::vector<int> maximum,
                        std::vector<std::pair<int, int>> &maximaTriplets,
                        std::vector<std::array<int, 46>> &saddleTriplets,
                        const double *offset,
                        const int data_size, const int globalMin);
    void computelargestSaddlesForMax(const int nMax, 
                                    const std::vector<std::array<int, 46>> &saddleTriplets, 
                                    std::unordered_map<int, int> &largestSaddlesForMax,
                                    const std::vector<int> *saddles2);
    void computesmallestSaddlesForMin(const int nMin, 
                                    const std::vector<std::array<int, 46>> &saddleTriplets, 
                                    std::unordered_map<int, int> &smallestSaddlesForMin,
                                    const std::vector<int> *minimum);
    // void computelargestSaddlesForMax(const int nMax, const std::vector<std::array<int, 46>> &saddleTriplets, std::unordered_map<int, int> &largestSaddlesForMax, const std::vector<int> *saddles2);
    void reorderVectorByTemparray(std::vector<std::pair<int, int>>& vec, const std::vector<int>& temparray);
    void get_wrong_index_saddles();
    void get_wrong_index_saddles_join();
    void saddle_loops(std::string type);
    void saddle_loops_join(std::string type);
    void get_wrong_index_min();
    void mappath(int* DS_M, int* AS_M, const double* offset);
    int computePathCompressionSingle(
        int *const segmentation,
        const bool computeAscending,
        const double *const offset) ;

}

#endif // EXTREM_H