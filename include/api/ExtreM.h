#ifndef EXTREM_H
#define EXTREM_H
#include <cstdint>
#include <stddef.h>
#include <string>
#include <array>
#include <vector>

// class UnionFind {

//     public:
//     UnionFind();
//     UnionFind(const UnionFind &other);

//     bool operator<(const UnionFind &other) const;
//     bool operator>(const UnionFind &other) const;

//     UnionFind *find();

//     int getRank() const;
//     void setParent(UnionFind *parent);
//     void setRank(const int &rank);

//     static UnionFind *makeUnion(UnionFind *uf0, UnionFind *uf1);
//     static UnionFind *makeUnion(std::vector<UnionFind *> &sets);

//     protected:
//     int rank_;
//     UnionFind *parent_;
// };

// UnionFind::UnionFind() {
//     rank_ = 0;
//     parent_ = this;
// }

// UnionFind::UnionFind(const UnionFind &other){
//     rank_ = other.rank_;
//     parent_ = this;
// }

// UnionFind *UnionFind::find() {
//     if (parent_ == this)
//         return this;
//     else {
//         parent_ = parent_->find();
//         return parent_;
//     }
// }

// void UnionFind::setParent(UnionFind *parent) {
//     parent_ = parent;
// }

// void UnionFind::setRank(const int &rank) {
//     rank_ = rank;
// }

// int UnionFind::getRank() const {
//     return rank_;
// }

// bool UnionFind::operator<(const UnionFind &other) const {
//     return rank_ < other.rank_;
// }

// bool UnionFind::operator>(const UnionFind &other) const {
//     return rank_ > other.rank_;
// }



// UnionFind *UnionFind::makeUnion(UnionFind *uf0, UnionFind *uf1) {
//     uf0 = uf0->find();
//     uf1 = uf1->find();

//     if (uf0 == uf1) {
//         return uf0;
//     } else if (uf0->getRank() > uf1->getRank()) {
//         uf1->setParent(uf0);
//         return uf0;
//     } else if (uf0->getRank() < uf1->getRank()) {
//         uf0->setParent(uf1);
//         return uf1;
//     } else {
//         uf1->setParent(uf0);
//         uf0->setRank(uf0->getRank() + 1);
//         return uf0;
//     }
// }

// UnionFind *UnionFind::makeUnion(std::vector<UnionFind *> &sets) {
//     UnionFind *n = nullptr;

//     if (!sets.size())
//         return nullptr;

//     if (sets.size() == 1)
//         return sets[0];

//     for (int i = 0; i < (int)sets.size() - 1; i++)
//         n = makeUnion(sets[i], sets[i + 1]);

//     return n;
// }

struct Branch {
    std::vector<std::pair<int, int>> vertices; // order, globalId, first pair is the maximum
    Branch *parentBranch = nullptr;
};

extern "C"{
    extern int directions[78], direction_to_index_mapping[26][3]; 
    extern double* d_deltaBuffer;
    extern int width, height, depth, maxNeighbors, num_Elements, count_f_max, count_f_min, count_f_saddle, number_of_false_cases;
    extern int wrong_max_counter, globalMin, dec_globalMin;
    extern int* adjacency, *dec_vertex_type, *vertex_type, *all_max, *all_min, *all_saddle;
    extern double *decp_data, *input_data;
    extern double bound;
    extern int *or_saddle_max_map, *wrong_neighbors, *wrong_neighbors_index, *wrong_rank_max, *wrong_rank_max_index;
    extern std::vector<std::array<int, 46>> saddleTriplets, dec_saddleTriplets;
    extern std::vector<std::vector<int>> vertex_cells;
    extern std::vector<int> saddles2, maximum, dec_saddles2, dec_maximum;


    
    void get_vertex_traingle();
    void getdata(const std::string &filename, 
                const double er, 
                int data_size);

    int ComputeDescendingManifold(const double *offset, 
                                  int *DS_M);
    int ComputeAscendingManifold(const double *offset, 
                                  int *AS_M);
    void classifyAllVertices(int *vertex_type, const double* heightMap, int type);
    void computeAdjacency();
    void c_loops();
    int classifyVertex(const int vertexId, const double *offset, 
                                        const int *desManifold, const int *ascManifold,
                                        int *types);

    int computeNumberOfLinkComponents(const int* linkVertices, const int nLinkVertices);
    void sortAndRemoveDuplicates(std::array<int, 46> &triplet);
    void computeCP(std::vector<std::pair<int, int>> &maximaTriplets, std::vector<std::array<int, 46>> &saddleTriplets, 
                    const double *offset, const int *desManifold, const int *ascManifold, std::vector<int> &saddles2, 
                    std::vector<int> &maximum, int &globalMin, int translation = 0);

    void getSortedIndexes(const double* vec, int* indexes);
    void findAscPaths(  std::vector<std::array<int, 46>> &saddleTriplets,
                        std::vector<int> &maximaLocalToGlobal,
                        std::vector<int> &saddlesLocalToGlobal,
                        const int *saddles2, const int *maximum,
                        const double *offset,
                        const int *desManifold, const int *ascManifold);

    int constructPersistencePairs(
                        std::vector<std::pair<int, int>> &pairs,
                        std::vector<std::pair<int, int>> &maximaTriplets,
                        std::vector<std::array<int, 46>> &saddleTriplets,
                        const int* maximum,
                        const int* saddles2,
                        const int nMax,
                        int globalMin);
    void compute_Max_for_Saddle(int saddle, const double *offset);
    void get_wrong_neighbors(int saddle, int &num_false_cases, const double *offset);
    int fixpath(int i, int direction);
    void get_false_criticle_points();
    void r_loops();
    void s_loops();
    void get_wrong_index_max();
    void compute_MergeT(std::vector<Branch> &branches, 
                        const std::vector<int> saddles2, 
                        const std::vector<int> maximum,
                        std::vector<std::pair<int, int>> &maximaTriplets,
                        std::vector<std::array<int, 46>> &saddleTriplets,
                        const double *offset,
                        const int data_size);
}

#endif // EXTREM_H