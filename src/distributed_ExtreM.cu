// MPI + CUDA with 3D block partitioning and ghost exchange for 14 neighbor directions

#include <mpi.h>
// #include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>
#include <algorithm>

#define WIDTH  64
#define HEIGHT 64
#define DEPTH  64
#define GHOST 1
// nvcc -ccbin mpicxx -std=c++17 distributed_ExtreM.cu -o ExtreM_distributed -lzstd
// srun --gpu-bind=closest --gpus-per-task=1 -n 4 ./ExtreM_distributed
struct GhostRequest {
    int requester_rank;
    int ghost_gid;
    int target;
    int position;

    bool operator<(const GhostRequest& other) const {
        if (requester_rank != other.requester_rank)
            return requester_rank < other.requester_rank;
        if (ghost_gid != other.ghost_gid)
            return ghost_gid < other.ghost_gid;
        return target < other.target;
    }

    bool operator==(const GhostRequest& other) const {
        return requester_rank == other.requester_rank &&
               ghost_gid == other.ghost_gid &&
               target == other.target;
    }

};
// 14 neighbor directions
__device__ int neighborOffsets[14][3] = {
    { 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0}, { 0, -1,  0},
    { 0,  0,  1}, { 0,  0, -1}, {-1,  1,  0}, { 1, -1,  0},
    { 0,  1,  1}, { 0, -1, -1}, {-1,  0,  1}, { 1,  0, -1},
    {-1,  1,  1}, { 1, -1, -1}
};

int host_neighborOffsets[14][3] = {
    { 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0}, { 0, -1,  0},
    { 0,  0,  1}, { 0,  0, -1}, {-1,  1,  0}, { 1, -1,  0},
    { 0,  1,  1}, { 0, -1, -1}, {-1,  0,  1}, { 1,  0, -1},
    {-1,  1,  1}, { 1, -1, -1}
};

__global__ void ComputeDirection(double* input_data, int* DS_M, int pitch_x, int pitch_y,
    int width, int height, int depth, int padded_z,
    int global_offset_x, int global_offset_y, int global_offset_z, int rank) {
    int px = threadIdx.x + blockIdx.x * blockDim.x;
    int py = threadIdx.y + blockIdx.y * blockDim.y;
    int pz = threadIdx.z + blockIdx.z * blockDim.z;

    if (px >= pitch_x || py >= pitch_y || pz >= padded_z) return;

    int center_idx = px + py * pitch_x + pz * pitch_x * pitch_y;
    
    // Compute global index of this vertex
    // int gx = px + global_offset_x;
    // int gy = py + global_offset_y;
    // int gz = pz + global_offset_z;
    // int global_id = gz * (HEIGHT * WIDTH) + gy * WIDTH + gx;

    // Check if this point is in the core region
    if (px >= 1 && px < pitch_x - 1 &&
        py >= 1 && py < pitch_y - 1 &&
        pz >= 1 && pz < padded_z - 1) {

        int largest_neighbor_idx = center_idx;
        // int largest_global_id = global_id;

        for (int j = 0; j < 14; ++j) {
                int dx = neighborOffsets[j][0];
                int dy = neighborOffsets[j][1];
                int dz = neighborOffsets[j][2];

                int nx = px + dx;
                int ny = py + dy;
                int nz = pz + dz;
                int n_idx = nx + ny * pitch_x + nz * pitch_x * pitch_y;
                if (nx >= 0 && nx < pitch_x && ny >= 0 && ny < pitch_y && nz >= 0 && nz < padded_z) {
                    

                    // int ngx = nx - 1 + global_offset_x;
                    // int ngy = ny - 1 + global_offset_y;
                    // int ngz = nz - 1 + global_offset_z;
                    // int neighbor_gid = ngz * (HEIGHT * WIDTH) + ngy * WIDTH + ngx;

                    if (input_data[largest_neighbor_idx] < input_data[n_idx] ||
                        (input_data[largest_neighbor_idx] == input_data[n_idx] && largest_neighbor_idx < n_idx)) {
                        largest_neighbor_idx = n_idx;
                        // largest_global_id = neighbor_gid;
                    }
                }
            }
        DS_M[center_idx] = largest_neighbor_idx;
        
    } else {
        // ghost cell: point to self
        DS_M[center_idx] = center_idx;
        
    }
}


void exchange_extended_ghost_layers(double* local_host, int padded_x, int padded_y, int padded_z,
                                    int local_x, int local_y, int local_z,
                                    MPI_Comm cart_comm, int rank) {
    MPI_Request reqs[28];
    int req_index = 0;

    auto index = [&](int x, int y, int z) {
        return x + y * padded_x + z * padded_x * padded_y;
    };

    for (int n = 0; n < 14; ++n) {
        int dx = host_neighborOffsets[n][0];
        int dy = host_neighborOffsets[n][1];
        int dz = host_neighborOffsets[n][2];

        int src_coords[3], dst_coords[3];
        MPI_Cart_coords(cart_comm, rank, 3, src_coords);
        dst_coords[0] = src_coords[0] + dx;
        dst_coords[1] = src_coords[1] + dy;
        dst_coords[2] = src_coords[2] + dz;

        int dst_rank;
        int status = MPI_Cart_rank(cart_comm, dst_coords, &dst_rank);
        if (status != MPI_SUCCESS) continue;

        // Determine send and recv locations
        // For simplicity, send/recv 1-layer slabs covering ghost area
        int send_x0 = (dx == -1) ? GHOST : (dx == 1 ? local_x + GHOST - 1 : GHOST);
        int send_y0 = (dy == -1) ? GHOST : (dy == 1 ? local_y + GHOST - 1 : GHOST);
        int send_z0 = (dz == -1) ? GHOST : (dz == 1 ? local_z + GHOST - 1 : GHOST);

        int recv_x0 = send_x0 + dx;
        int recv_y0 = send_y0 + dy;
        int recv_z0 = send_z0 + dz;

        int slab_size = 1;
        if (dx == 0) slab_size *= local_x;
        if (dy == 0) slab_size *= local_y;
        if (dz == 0) slab_size *= local_z;

        MPI_Isend(&local_host[index(send_x0, send_y0, send_z0)], slab_size, MPI_DOUBLE, dst_rank, 100 + n, cart_comm, &reqs[req_index++]);
        MPI_Irecv(&local_host[index(recv_x0, recv_y0, recv_z0)], slab_size, MPI_DOUBLE, dst_rank, 100 + n, cart_comm, &reqs[req_index++]);
    }

    MPI_Waitall(req_index, reqs, MPI_STATUSES_IGNORE);
}


void gather_DS_M_results(int* device_DS_M, int* host_DS_M, int padded_x, int padded_y, int padded_z,
    int local_x, int local_y, int local_z,
    int rank, int size, MPI_Comm comm, int global_offset_x,
    int global_offset_y,
    int global_offset_z) {
        int core_size = local_x * local_y * local_z;
        int* local_core = new int[core_size];
        int index = 0;

        for (int z = GHOST; z < local_z + GHOST; ++z) {
            for (int y = GHOST; y < local_y + GHOST; ++y) {
                for (int x = GHOST; x < local_x + GHOST; ++x) {
                    int idx = x + y * padded_x + z * padded_x * padded_y;
                    int result = host_DS_M[idx];
                    int localx = result % padded_x;
                    int localy = (result / padded_x) % padded_y;
                    int localz = (result / (padded_x * padded_z)) % padded_z;
                    int global_id = (localx + global_offset_x) + (localy+global_offset_y) * WIDTH + (localz+global_offset_z) * WIDTH * HEIGHT;

                    local_core[index++] = global_id;
                }
            }
        }

        int* global_result = nullptr;
        if (rank == 0) global_result = new int[core_size * size];

        MPI_Gather(local_core, core_size, MPI_INT,
            global_result, core_size, MPI_INT,
            0, comm);

        if (rank == 0) {
            printf("[Rank 0] Gathered global DS_M results:\n");
            // for (int i = 0; i < core_size * size; ++i) {
            //     printf("Vertex %d -> %d\n", i, global_result[i]);
            // }
            delete[] global_result;
        }

        delete[] local_core;
}

__global__ void PathCompression(int* DS_M, int total_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;

    int cur = DS_M[i];
    while (DS_M[cur] != cur) {
        cur = DS_M[cur];
    }
    DS_M[i] = cur;
}


__global__ void GhostPathCompression(GhostRequest* DS_M, int total_elements, int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;
    // printf("%d \n", i);
    int cur = DS_M[i].target;
    if(rank == 0) printf("%d \n", i);
    int id = DS_M[cur].position;
    
    while (DS_M[id].target != cur) {
        cur = DS_M[cur].target;
        id = DS_M[cur].position;
    }
    DS_M[i].target = cur;
}



__global__ void MarkGhostDependencies(int* DS_M, int* output_gid_array,
    int padded_x, int padded_y, int padded_z,
    int global_offset_x, int global_offset_y, int global_offset_z,
    int local_gx_min, int local_gy_min, int local_gz_min,
    int local_gx_max, int local_gy_max, int local_gz_max) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = padded_x * padded_y * padded_z;
    if (i >= total) return;

    int px = i % padded_x;
    int py = (i / padded_x) % padded_y;
    int pz = i / (padded_x * padded_y);

    if (px < GHOST || px >= padded_x - GHOST ||
        py < GHOST || py >= padded_y - GHOST ||
        pz < GHOST || pz >= padded_z - GHOST)
        return; // only core region

    int target_gid = DS_M[i];
    int tx = target_gid % WIDTH;
    int ty = (target_gid / WIDTH) % HEIGHT;
    int tz = target_gid / (WIDTH * HEIGHT);

    if (tx < local_gx_min || tx >= local_gx_max ||
        ty < local_gy_min || ty >= local_gy_max ||
        tz < local_gz_min || tz >= local_gz_max) {
        output_gid_array[i] = tx + global_offset_x + (ty + global_offset_y) * WIDTH + (tz + global_offset_z) * WIDTH * HEIGHT;  // mark ghost dependency
    } else {
        output_gid_array[i] = -1;
    }
}

void gather_ghost_flags_to_rank0(int* device_flags, int total_elements, int rank, int size, int* gathered_flags_root) {
    // CUDA-aware MPI assumed: each rank sends its device_flags directly
    // gathered_flags_root must be allocated on rank 0: size * total_elements ints
    if (rank == 0) {
        printf("Rank %d: total_elements = %d, device_flags = %p\n", rank, total_elements, device_flags);
        MPI_Gather(MPI_IN_PLACE, total_elements, MPI_INT,
                   gathered_flags_root, total_elements, MPI_INT,
                   0, MPI_COMM_WORLD);
    } else {
        printf("Rank %d: total_elements = %d, device_flags = %p\n", rank, total_elements, device_flags);

        MPI_Gather(device_flags, total_elements, MPI_INT,
                   nullptr, 0, MPI_INT,
                   0, MPI_COMM_WORLD);
    }
}

int get_owner_rank(int gid, int width, int height, int depth,
    int px, int py, int pz) {
    int gx = gid % width;
    int gy = (gid / width) % height;
    int gz = gid / (width * height);

    int block_x = width / px;
    int block_y = height / py;
    int block_z = depth / pz;

    int cx = gx / block_x;
    int cy = gy / block_y;
    int cz = gz / block_z;

    return cz * (py * px) + cy * px + cx;
}

__device__ int get_gid_from_local_index_device(int local_idx,
    int padded_x, int padded_y,
    int global_offset_x,
    int global_offset_y,
    int global_offset_z) {

    int lz = local_idx / (padded_x * padded_y);
    int ly = (local_idx / padded_x) % padded_y;
    int lx = local_idx % padded_x;

    int gx = lx + global_offset_x;
    int gy = ly + global_offset_y;
    int gz = lz + global_offset_z;

    return gx + gy * WIDTH + gz * WIDTH * HEIGHT;
}


__device__ int get_local_index_from_gid_device(int gid,
    int width, int height,
    int global_offset_x,
    int global_offset_y,
    int global_offset_z,
    int padded_x, int padded_y) {

    int gx = gid % width;
    int gy = (gid / width) % height;
    int gz = gid / (width * height);

    int lx = gx - global_offset_x + GHOST;
    int ly = gy - global_offset_y + GHOST;
    int lz = gz - global_offset_z + GHOST;

    return lx + ly * padded_x + lz * padded_x * padded_y;
}


__global__ void FetchGhostTargetsKernel(GhostRequest* device_requests, int num_requests,
    const int* device_DS_M,
    int width, int height,
    int global_offset_x, int global_offset_y, int global_offset_z,
    int padded_x, int padded_y) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_requests) return;

    // GhostRequest& s = device_requests[i];
    GhostRequest temp = device_requests[i];

    int local_idx = get_local_index_from_gid_device(
                    temp.ghost_gid,
                    width, height,
                    global_offset_x, global_offset_y, global_offset_z,
                    padded_x, padded_y);
    temp.position = i;
    temp.target = get_gid_from_local_index_device(device_DS_M[local_idx],
                                                padded_x, padded_y,
                                                global_offset_x,
                                                global_offset_y,
                                                global_offset_z);  
    device_requests[i] = temp;
}

std::vector<int> get_cartesian_neighbors(MPI_Comm cart_comm, int rank) {
    std::vector<int> neighbors;
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int neighbor_coords[3] = {coords[0] + dx, coords[1] + dy, coords[2] + dz};
                int neighbor_rank;
                if (MPI_Cart_rank(cart_comm, neighbor_coords, &neighbor_rank) == MPI_SUCCESS) {
                    neighbors.push_back(neighbor_rank);
                }
            }
        }
    }
    return neighbors;
}

__device__ int get_owner_rank_gpu(int gid, int width, int height, int depth,
    int px, int py, int pz,
    int* dims) {
    int gx = gid % width;
    int gy = (gid / width) % height;
    int gz = gid / (width * height);

    int block_x = width / px;
    int block_y = height / py;
    int block_z = depth / pz;

    int cx = gx / block_x;
    int cy = gy / block_y;
    int cz = gz / block_z;

    return cx + cy * px + cz * px * py; // assuming row-major rank layout
}

__global__ void build_requests_kernel(const int* ghost_flags, GhostRequest* output,
    int* request_offsets, int width, int height, int depth,
    int px, int py, int pz, int* dims, int total_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;

    int gid = ghost_flags[i];
    if (gid == -1) return;

    int rank = get_owner_rank_gpu(gid, width, height, depth, px, py, pz, dims);
    int offset = atomicAdd(&request_offsets[rank], 1);
    output[rank * 1024 + offset] = {gid, -1}; // 1024 = max per rank
}

void exchange_ghosts_cuda(const std::map<int, GhostRequest*>& device_requests_to_send,
    const std::map<int, int>& send_counts,
    MPI_Comm cart_comm,
    std::vector<GhostRequest*>& device_received_buffers,
    std::vector<int>& recv_counts) {
        int rank;
        MPI_Comm_rank(cart_comm, &rank);
        std::vector<int> neighbors = get_cartesian_neighbors(cart_comm, rank);

        std::vector<MPI_Request> send_requests, recv_requests;
        device_received_buffers.resize(neighbors.size());
        recv_counts.resize(neighbors.size());

        const int max_expected = 1024;  // upper bound for recv count (adjustable)

        for (size_t i = 0; i < neighbors.size(); ++i) {
            int nbr = neighbors[i];

            // Allocate device recv buffer
            GhostRequest* device_recv_buf;
            cudaMalloc(&device_recv_buf, max_expected * sizeof(GhostRequest));
            device_received_buffers[i] = device_recv_buf;
            recv_counts[i] = max_expected;

            // Post non-blocking recv directly into GPU memory
            recv_requests.emplace_back();
            MPI_Irecv(device_recv_buf, max_expected * sizeof(GhostRequest), MPI_BYTE,
            nbr, 0, cart_comm, &recv_requests.back());

            // Post non-blocking send from device buffer
            auto it = device_requests_to_send.find(nbr);
            if (it != device_requests_to_send.end()) {
                GhostRequest* send_buf = it->second;
                int count = send_counts.at(nbr);
                send_requests.emplace_back();
                MPI_Isend(send_buf, count * sizeof(GhostRequest), MPI_BYTE,
                nbr, 0, cart_comm, &send_requests.back());
            }
        }

        MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
} // buffers are now available on GPU

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);

    int coords[3];
    MPI_Comm cart_comm;
    int periods[3] = {1, 1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    int px = dims[0], py = dims[1], pz = dims[2];
    int cx = coords[0], cy = coords[1], cz = coords[2];

    int local_x = WIDTH / px;
    int local_y = HEIGHT / py;
    int local_z = DEPTH / pz;
    
    // 数据：都在 padded 区域里
    double* device_input;  // padded 的 input_data，在 device 上
    int* device_DS_M;      // padded 的 DS_M，在 device 上
    int width = WIDTH, height = HEIGHT, depth = DEPTH;
    int padded_x = local_x + 2 * GHOST;
    int padded_y = local_y + 2 * GHOST;
    int padded_z = local_z + 2 * GHOST;
    size_t local_bytes = padded_x * padded_y * padded_z * sizeof(double);

    int total_elements = padded_x * padded_y * padded_z;
    cudaError_t err = cudaMalloc(&device_DS_M, total_elements * sizeof(int));
    

    double* local_host = (double*)malloc(local_bytes);
    memset(local_host, 0, local_bytes);
    double* input_data = nullptr;

    if (rank == 0) {
        input_data = (double*)malloc(WIDTH * HEIGHT * DEPTH * sizeof(double));
        for (int i = 0; i < WIDTH * HEIGHT * DEPTH; ++i) input_data[i] = i;
    }

    MPI_Datatype subarray;
    int sizes[3] = {DEPTH, HEIGHT, WIDTH};
    int subsizes[3] = {local_z, local_y, local_x};
    int starts[3] = {cz * local_z, cy * local_y, cx * local_x};
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray);
    MPI_Type_commit(&subarray);

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            int r_coords[3];
            MPI_Cart_coords(cart_comm, i, 3, r_coords);
            int r_starts[3] = {r_coords[2] * local_z, r_coords[1] * local_y, r_coords[0] * local_x};
            MPI_Type_create_subarray(3, sizes, subsizes, r_starts, MPI_ORDER_C, MPI_DOUBLE, &subarray);
            MPI_Type_commit(&subarray);
            if (i == 0) {
                memcpy(&local_host[GHOST * padded_x * padded_y + GHOST * padded_x + GHOST], input_data, local_x * local_y * local_z * sizeof(double));
            } else {
                MPI_Send(input_data, 1, subarray, i, 0, cart_comm);
            }
        }
    } else {
        MPI_Recv(&local_host[GHOST * padded_x * padded_y + GHOST * padded_x + GHOST], local_x * local_y * local_z, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
    }

    exchange_extended_ghost_layers(local_host, padded_x, padded_y, padded_z, local_x, local_y, local_z, cart_comm, rank);

    

    cudaMalloc(&device_input, local_bytes);
    
    cudaMemcpy(device_input, local_host, local_bytes, cudaMemcpyHostToDevice);

    dim3 block(8, 8, 4);
    dim3 grid((padded_x + block.x - 1) / block.x,
              (padded_y + block.y - 1) / block.y,
              (padded_z + block.z - 1) / block.z);

    int global_offset_x = cx * local_x;
    int global_offset_y = cy * local_y;
    int global_offset_z = cz * local_z;

    int local_gx_min = global_offset_x;
    int local_gy_min = global_offset_y;
    int local_gz_min = global_offset_z;
    int local_gx_max = global_offset_x + local_x;
    int local_gy_max = global_offset_y + local_y;
    int local_gz_max = global_offset_z + local_z;


    ComputeDirection<<<grid, block>>>(
        device_input,
        device_DS_M,
        padded_x,
        padded_y,
        local_x,
        local_y,
        local_z,
        padded_z,
        global_offset_x,
        global_offset_y,
        global_offset_z,
        rank
    );
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed compute direction: %s\n", rank, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    PathCompression<<<gridSize, blockSize>>>(device_DS_M, total_elements);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed path compression: %s\n", rank, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    cudaDeviceSynchronize();

    int *device_ghost_gid_flags;
    err = cudaMalloc(&device_ghost_gid_flags, total_elements * sizeof(int));
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed: %s\n", rank, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    cudaDeviceSynchronize();

    MarkGhostDependencies<<<gridSize, blockSize>>>(
        device_DS_M, device_ghost_gid_flags,
        padded_x, padded_y, padded_z,
        global_offset_x, global_offset_y, global_offset_z,
        local_gx_min, local_gy_min, local_gz_min,
        local_gx_max, local_gy_max, local_gz_max);
    cudaDeviceSynchronize();

    int* gathered_flags_root = nullptr;
    if (rank == 0) {
        gathered_flags_root = (int*) malloc(size * total_elements * sizeof(int));
        if (!gathered_flags_root) {
            fprintf(stderr, "malloc failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int total_ranks = px * py * pz;
    GhostRequest* device_requests;
    cudaMalloc(&device_requests, total_ranks * 1024 * sizeof(GhostRequest)); // max 1024 per rank

    int* device_request_offsets;
    cudaMalloc(&device_request_offsets, total_ranks * sizeof(int));
    cudaMemset(device_request_offsets, 0, total_ranks * sizeof(int));

    int* device_dims;
    cudaMalloc(&device_dims, 3 * sizeof(int));
    int dims_host[3] = {px, py, pz};
    cudaMemcpy(device_dims, dims_host, 3 * sizeof(int), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
      
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    build_requests_kernel<<<blocks, threads>>>(
        device_ghost_gid_flags, device_requests, device_request_offsets,
        width, height, depth, px, py, pz, device_dims, total_elements);
    cudaDeviceSynchronize();

    std::map<int, GhostRequest*> device_requests_to_send;
    std::map<int, int> send_counts;

    std::vector<int> host_offsets(total_ranks);
    cudaMemcpy(host_offsets.data(), device_request_offsets, total_ranks * sizeof(int), cudaMemcpyDeviceToHost);

    for (int r = 0; r < total_ranks; ++r) {
        if (host_offsets[r] > 0) {
            device_requests_to_send[r] = device_requests + r * 1024;
            send_counts[r] = host_offsets[r];
        }
    }

    std::vector<GhostRequest*> device_received_buffers;
    std::vector<int> recv_counts;

    exchange_ghosts_cuda(device_requests_to_send, send_counts, cart_comm,
                        device_received_buffers, recv_counts);
    
    cudaDeviceSynchronize();
    return 0;
    // int number_of_gv = 0;
    // std::vector<GhostRequest> gv;

    // std::vector<GhostRequest> receivedRequests;
    // GhostRequest* device_received;
    // int count_host = 0;
    // int total_resolved_ghosts = 0;
    // if(rank==0){
    //     for(int i=0;i<size * total_elements;i++){
    //         int id = i;
    //         if(gathered_flags_root[id] != -1){
    //             number_of_gv++;
    //             int owner_rank = get_owner_rank(gathered_flags_root[id], width, height, depth,
    //                                             px, py, pz);
                
    //             GhostRequest s{owner_rank,gathered_flags_root[id],-1};
    //             gv.push_back(s);
    //         }
            
    //     }
    //     std::sort(gv.begin(), gv.end());
    //     gv.erase(std::unique(gv.begin(), gv.end()), gv.end());
    //     for(int i = 0; i < gv.size(); i++){
    //         auto s = gv[i];
    //         // s.position = i;
    //     }

    //     std::vector<std::vector<GhostRequest>> neededPerRanks(size);

    //     total_resolved_ghosts = gv.size();
        
        
    //     for(auto s:gv){
    //         neededPerRanks[s.requester_rank].push_back(s);
    //     }
        
    //     for (int i = 0; i < size; ++i) {
    //         const auto& requests = neededPerRanks[i];
    //         int count = requests.size();
    
    //         if (i == 0) {
    //             count_host = count;
    //             cudaMalloc(&device_received, count * sizeof(GhostRequest));
    //             cudaMemcpy(device_received, requests.data(), count * sizeof(GhostRequest), cudaMemcpyHostToDevice);
    //         } else {

    //             GhostRequest* device_requests;
    //             cudaMalloc(&device_requests, count * sizeof(GhostRequest));
    //             cudaMemcpy(device_requests, requests.data(), count * sizeof(GhostRequest), cudaMemcpyHostToDevice);
                
    //             MPI_Send(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    //             MPI_Send(device_requests, count * sizeof(GhostRequest), MPI_BYTE, i, 1, MPI_COMM_WORLD);
    //         }
    //     }
    // }
    // else {
        
    //     MPI_Recv(&count_host, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     cudaMalloc(&device_received, count_host * sizeof(GhostRequest));
    //     MPI_Recv(device_received, count_host * sizeof(GhostRequest), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     // MPI_Recv(receivedRequests.data(), count * sizeof(GhostRequest), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }
    // cudaDeviceSynchronize();

    // printf("rank %d: %d\n", rank, count_host);
    // FetchGhostTargetsKernel<<<gridSize, blockSize>>>(device_received, count_host,
    //                                                     device_DS_M,
    //                                                     width, height,
    //                                                     global_offset_x, global_offset_y, global_offset_z,
    //                                                     padded_x, padded_y);
    
    // cudaDeviceSynchronize();

    // MPI_Bcast(&total_resolved_ghosts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // cudaDeviceSynchronize();

    // int sendcount = count_host; // 每个 rank 本地的不一样
    // std::vector<int> recvcounts(size);

    // MPI_Allgather(&sendcount, 1, MPI_INT,
    //             recvcounts.data(), 1, MPI_INT,
    //             MPI_COMM_WORLD);
    
    // std::vector<int> displs(size, 0);
    // for (int i = 1; i < size; ++i) {
    //     displs[i] = displs[i - 1] + recvcounts[i - 1];
    // }

    // for (int i = 0; i < size; ++i) {
    //     recvcounts[i] *= sizeof(GhostRequest);
    //     displs[i] *= sizeof(GhostRequest);
    // }
    
    // int total_recv = displs[size - 1] + recvcounts[size - 1];
    // cudaDeviceSynchronize();

    // GhostRequest* device_all_resolved;
    // cudaMalloc(&device_all_resolved, total_recv * sizeof(GhostRequest));
    // cudaDeviceSynchronize();
    
    // printf("rank: %d %d sendcount: %d\n", rank, count_host, sendcount);
    
    // MPI_Allgather(device_received, count_host, MPI_INT, allValuesFromRanks, count, MPI_INT, MPI_COMM_WORLD);
    // MPI_Allgatherv(
    //         device_received, sendcount * sizeof(GhostRequest), MPI_BYTE,
    //         device_all_resolved, recvcounts.data(), displs.data(), MPI_BYTE,
    //         MPI_COMM_WORLD);
    
   
    // GhostRequest *host_received = new GhostRequest[total_recv];
    // cudaMemcpy(host_received, device_all_resolved, total_recv*sizeof(GhostRequest),cudaMemcpyDeviceToHost);
    // // cudaMemcpy(host_DS_M, device_DS_M, total_elements * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i< total_recv; i++){
    //     auto s = host_received[i];
    //     if(s.ghost_gid!=0) printf("id: %d, target: %d, position: %d\n", s.ghost_gid, s.target, s.position);
    // }

    // cudaDeviceSynchronize();
    // // return 0;
    
    // int gridSize_Ghost = (total_recv + blockSize - 1) / blockSize;
    // GhostPathCompression<<<gridSize_Ghost, blockSize>>>(device_all_resolved, total_recv, rank);
    // cudaDeviceSynchronize();
    
    // return 0;
    // int  max_expected = 1024;
    // GhostRequest* device_recv_buffer;
    // cudaMalloc(&device_recv_buffer, max_expected * sizeof(GhostRequest));
    // recv_buffers.push_back(device_recv_buffer);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed gather: %s\n", rank, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int* host_DS_M = new int[total_elements];
    // cudaMemcpy(local_host, device_input, local_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_DS_M, device_DS_M, total_elements * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gather_DS_M_results(device_DS_M, host_DS_M,
                        padded_x, padded_y, padded_z,
                        local_x, local_y, local_z,
                        rank, size, cart_comm, global_offset_x,
                        global_offset_y,
                        global_offset_z);

    cudaDeviceSynchronize();



    cudaFree(device_input);

    free(local_host);
    if (rank == 0) free(input_data);
    MPI_Finalize();


    return 0;
}

