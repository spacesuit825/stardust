// C++
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>


// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

// CUDA
#include <vector_types.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvfunctional>
#include <cuda_runtime_api.h>

// CUB
//#include <cub/device/device_radix_sort.cuh>

#include <../src/engine/cuda/collision_detection.cuh>
#include <../src/engine/cuda/cuda_utils.hpp>
#include <../src/engine/cuda/cuda_helpers.cuh>

#include <iostream>

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

void debugVector(thrust::host_vector<float4>& vector) {
    std::cout << "\nDebug Vector Print\n" << "----------------------------\n";
    for (int i = 0; i < vector.size(); i++) {
        std::cout << "Vector " << i << ": ";
        std::cout << "[" << vector[i].x << ", " << vector[i].y << ", " << vector[i].z << "]\n";
    }
}

void debugList(std::string name, thrust::device_vector<float>& vector) {
    std::cout << "\n" << name << "\n" << "----------------------------\n";
    std::cout << "[";
    for (int i = 0; i < vector.size(); i++) {
        std::cout << vector[i] << " ";
    }
    std::cout << "]\n";
}

void debugList(std::string name, thrust::device_vector<int>& vector) {
    std::cout << "\n" << name << "\n" << "----------------------------\n";
    std::cout << "[";
    for (int i = 0; i < vector.size(); i++) {
        std::cout << vector[i] << " ";
    }
    std::cout << "]\n";
}

void computeAABB(
    thrust::host_vector<float4>& position, 
    thrust::host_vector<float>& radius, 
    thrust::host_vector<float4>& lower_bound, 
    thrust::host_vector<float4>& upper_bound) 
{
    for (int i = 0; i < position.size(); i++) {
        float4 pos = position[i];
        float r = radius[i];

        upper_bound[i] = make_float4(pos.x + r, pos.y + r, pos.z + r, 0.0f);
        lower_bound[i] = make_float4(pos.x - r, pos.y - r, pos.z - r, 0.0f);
    }
}

struct project_functorx
{
    __host__ __device__
        float operator()(const float4& x) const
    {
        return x.x;
    }
};

struct project_functory
{
    __host__ __device__
        float operator()(const float4& x) const
    {
        return x.x;
    }
};

struct project_functorz
{
    __host__ __device__
        float operator()(const float4& x) const
    {
        return x.x;
    }
};

void radixSortCustom(
    uint32_t* keys_in,
    uint32_t* values_in,
    uint32_t* keys_out,
    uint32_t* values_out,
    uint32_t* radices,
    uint32_t* radix_sums,
    int n
)
{

    // Note: this only works for positive floats for now
    STARDUST::SpatialPartition::sortCollisionList(
        keys_in,
        values_in,
        keys_out,
        values_out,
        radices,
        radix_sums,
        n
    );

}

void projectAABBx(
    thrust::device_vector<float4>& lower_bound,
    thrust::device_vector<float4>& upper_bound,
    thrust::device_vector<float>& lower,
    thrust::device_vector<float>& upper)
{
    thrust::transform(lower_bound.begin(), lower_bound.end(), lower.begin(), project_functorx());
    thrust::transform(upper_bound.begin(), upper_bound.end(), upper.begin(), project_functorx());
}

void projectAABBy(
    thrust::device_vector<float4>& lower_bound,
    thrust::device_vector<float4>& upper_bound,
    thrust::device_vector<float>& lower,
    thrust::device_vector<float>& upper)
{
    thrust::transform(lower_bound.begin(), lower_bound.end(), lower.begin(), project_functory());
    thrust::transform(upper_bound.begin(), upper_bound.end(), upper.begin(), project_functory());
}

void projectAABBz(
    thrust::device_vector<float4>& lower_bound,
    thrust::device_vector<float4>& upper_bound,
    thrust::device_vector<float>& lower,
    thrust::device_vector<float>& upper)
{
    thrust::transform(lower_bound.begin(), lower_bound.end(), lower.begin(), project_functorz());
    thrust::transform(upper_bound.begin(), upper_bound.end(), upper.begin(), project_functorz());
}

void radixSort(
    float* lower,
    int* idx,
    int n_objects)
{
    thrust::sort_by_key(thrust::device, lower, lower + n_objects, idx);
}

__device__ void populateCollisions(
    int tid, 
    int& collision_length, 
    int* pending_collisions, 
    int& idx)
{

    bool unique_collision = true;
    for (int k = 0; k < 10; k++) { // <-- Max collisions is 10!!
        if (idx == pending_collisions[k]) {

            unique_collision = false;
        }
    }

    if (unique_collision == false) {
        return;
    }
    else {
        
        pending_collisions[collision_length + 1] = idx;
        collision_length += 1;

        return;
    }
}

__global__ void sweepBlocks(
    float* upperx,
    float* lowerx,
    float* uppery,
    float* lowery,
    float* upperz,
    float* lowerz,
    int* idxx,
    int* idxy,
    int* idxz,
    int* potential_collision,
    int n_objects,
    int padding)
{
    // Shared memory
    extern __shared__ int collisions[];

    int obj_idx = blockIdx.x;
    int sorted_home_idx = idxx[obj_idx];
    float home_upper_extent = upperx[sorted_home_idx];

    int phantom_idx;
    float phantom_lower_extent;

    int phantom_position = threadIdx.x + blockIdx.x;

    if (phantom_position >= n_objects) {
        return;
    }

    if (phantom_position == sorted_home_idx) {
        return;
    }

    phantom_idx = idxx[phantom_position];

    phantom_lower_extent = lowerx[phantom_position];

    

    // Check X proj
    if (phantom_lower_extent <= home_upper_extent) { // <-- TODO: change this so it starts with axis with most position variance

        home_upper_extent = uppery[sorted_home_idx];
        phantom_lower_extent = lowery[phantom_idx];

        // Check Y proj
        if (phantom_lower_extent <= home_upper_extent) {

            home_upper_extent = upperz[sorted_home_idx];
            phantom_lower_extent = lowerz[phantom_idx];

            // Check Z proj
            if (phantom_lower_extent <= home_upper_extent) {

                //printf("collision detected between %d and %d\n", sorted_home_idx, phantom_idx);

            }
        }
    }
}

__global__ void sweep(
    float* upperx,
    float* lowerx,
    float* uppery,
    float* lowery,
    float* upperz,
    float* lowerz,
    int* idxx,
    int* idxy,
    int* idxz,
    int* potential_collision,
    int n_objects,
    int padding)
{

    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= n_objects) {
        return;
    }

    int* idx;
    float* upper;
    float* lower;
    int* coll;

    int collision = 0;

    int pending_collisions[10];

    int home_idx = tid;
    int sorted_home_idx = idxx[home_idx];
    float home_upper_extent = upperx[sorted_home_idx];

    int phantom_idx;
    float phantom_lower_extent;

    int pending_collision_length = 0;

    int n_tid = tid + 64;
    if (n_tid > n_objects) {
        n_tid = n_objects;
    }
    else {
        n_tid = 10;
    }

    for (int i = tid + 1; i < n_tid; i++) {

        if (i == sorted_home_idx) {
            continue;
        }

        phantom_lower_extent = lowerx[i];

        phantom_idx = idxx[i];

        // Check X proj
        if (phantom_lower_extent <= home_upper_extent) { // <-- TODO: change this so it starts with axis with most position variance

            home_upper_extent = uppery[sorted_home_idx];
            phantom_lower_extent = lowery[phantom_idx];

            // Check Y proj
            if (phantom_lower_extent <= home_upper_extent) {

                home_upper_extent = upperz[sorted_home_idx];
                phantom_lower_extent = lowerz[phantom_idx];

                // Check Z proj
                if (phantom_lower_extent <= home_upper_extent) {

                    populateCollisions(tid, pending_collision_length, potential_collision + tid * padding, phantom_idx);
                    //printf("Collision detected between: %d and %d\n", sorted_home_idx, phantom_idx);

                }
            }
        }
    }

}

__global__ void projectAxis(
    int n_toic,
    int* d_obj_to_cells_ptr,
    int* d_cells_ptr,
    int* d_cells_occ_ptr,
    int* d_cell_prefix_ptr,
    float4 global_lower,
    float4 global_upper,
    float4* d_lower_bound_ptr,
    float4* d_upper_bound_ptr,
    float* d_lowerx_ptr,
    float* d_upperx_ptr,
    float* d_lowery_ptr,
    float* d_uppery_ptr,
    float* d_lowerz_ptr,
    float* d_upperz_ptr,
    int n_objects)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= n_toic) {
        return;
    }

    float domain_size = global_upper.x - global_lower.x;

    int cell_idx = d_obj_to_cells_ptr[tid];
    
    float4 lower_bound = d_lower_bound_ptr[tid];
    float4 upper_bound = d_upper_bound_ptr[tid];

    d_lowerx_ptr[tid] = lower_bound.x + cell_idx * domain_size;
    d_lowery_ptr[tid] = lower_bound.y;
    d_lowerz_ptr[tid] = lower_bound.z;

    d_upperx_ptr[tid] = upper_bound.x + cell_idx * domain_size;
    d_uppery_ptr[tid] = upper_bound.y;
    d_upperz_ptr[tid] = upper_bound.z;
}

struct isValid
{
    __host__ __device__
        bool operator()(const int& value) const
    {
        return value >= 0;
    }
};

float randFloat(float a, float b) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

__global__ void mapObjectsToCells(
    int n_objects, 
    int* idxx, 
    float4* lower_bound, 
    float4* upper_bound, 
    float4 global_lower, 
    float4 global_upper, 
    int cell_count, 
    int* cells,
    int padding,
    int* d_cell_occ_ptr,
    int* d_cell_prefix_ptr
)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= n_objects) {
        return;
    }

    float cell_size = (global_upper.x - global_lower.x) / cell_count;

    float4 lower = lower_bound[tid];
    float4 upper = upper_bound[tid];

    int lower_y = (int)floorf((lower.y - global_lower.y) / cell_size);
    int lower_z = (int)floorf((lower.z - global_lower.z) / cell_size);

    int upper_y = (int)ceilf((upper.y - global_lower.y) / cell_size);
    int upper_z = (int)ceilf((upper.z - global_lower.z) / cell_size);

    for (int y = lower_y; y < upper_y; ++y) {
        for (int z = lower_z; z < upper_z; ++z) {

            int idx = (z * cell_count) + y;
            int spacing = padding * idx;

            int loc = atomicAdd(&d_cell_occ_ptr[idx], 1);
            cells[spacing + loc] = tid;
        }
    }
}

__global__ void reduceCellArray(
    int n_cells,
    int n_overlaps,
    int* d_cells_prefix_ptr,
    int* d_cells_occ_ptr,
    int* d_temp_cells_ptr,
    int* d_cells_ptr,
    int padding,
    int* d_obj_to_cells_ptr
)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= n_overlaps) {
        return;
    }

    int cell_idx = (int)floorf(tid / padding);
    int idx = tid;

    int cell_occupation = d_cells_occ_ptr[cell_idx];

    int overlaps = cell_occupation;

    int cell_pos = idx - cell_idx * padding;

    

    if (cell_pos >= cell_occupation) {
        return;
    }

    int cell_prefix = d_cells_prefix_ptr[cell_idx];
    int obj = d_temp_cells_ptr[idx];

    printf("Cell pos: %d\n", cell_prefix + cell_pos);

    d_cells_ptr[cell_prefix + cell_pos] = obj;
    d_obj_to_cells_ptr[cell_prefix + cell_pos] = cell_idx;
}



__global__ void prefixSumScan(int* output, int* input, int n, int powerOfTwo)
{
    extern __shared__ int temp[];// allocated on invocation
    int threadID = threadIdx.x;

    int ai = threadID;
    int bi = threadID + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


    if (threadID < n) {
        temp[ai + bankOffsetA] = input[ai];
        temp[bi + bankOffsetB] = input[bi];
    }
    else {
        temp[ai + bankOffsetA] = 0;
        temp[bi + bankOffsetB] = 0;
    }


    int offset = 1;
    for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (threadID == 0) {
        temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
    }

    for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (threadID < n) {
        output[ai] = temp[ai + bankOffsetA];
        output[bi] = temp[bi + bankOffsetB];
    }
}

int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

void launchPrefixScan(int* output, int* input, int length) {
    int powerOfTwo = nextPowerOfTwo(length);

    prefixSumScan << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (
        output,
        input,
        length,
        powerOfTwo
        );
}

int main() {

    int n_objects = 2;
    int max_collisions = 10;

    int particles_per_cell = 5;

    float4 global_upper = make_float4(1.0, 1.0, 1.0, 0.0);
    float4 global_lower = make_float4(0.0, 0.0, 0.0, 0.0);

    thrust::host_vector<int> total_objs_in_cells(1);
    thrust::host_vector<float4> position(n_objects);
    thrust::host_vector<float> radius(n_objects);
    thrust::host_vector<float4> lower_bound(n_objects);
    thrust::host_vector<float4> upper_bound(n_objects);

    thrust::host_vector<float> lowerx(n_objects);
    thrust::host_vector<float> upperx(n_objects);
    thrust::host_vector<float> lowery(n_objects);
    thrust::host_vector<float> uppery(n_objects);
    thrust::host_vector<float> lowerz(n_objects);
    thrust::host_vector<float> upperz(n_objects);

    thrust::host_vector<int> idxx(n_objects);
    thrust::host_vector<int> idxy(n_objects);
    thrust::host_vector<int> idxz(n_objects);

    thrust::host_vector<int> sorted_idxx(n_objects);
    thrust::host_vector<float> sorted_lowerx(n_objects);

    thrust::host_vector<int> radix(NUM_BLOCKS * NUM_RADICES * GROUPS_PER_BLOCK);
    thrust::host_vector<int> radix_sum(NUM_RADICES);

    thrust::host_vector<int> potential_collision(max_collisions * n_objects + max_collisions);

    /*for (int i = 0; i < n_objects; i++) {
        position[i] = make_float4(randFloat(global_lower.x + 0.5, global_upper.x + 0.5), randFloat(global_lower.y + 0.5, global_upper.y + 0.5), randFloat(global_lower.z + 0.5, global_upper.z + 0.5), 0.0f);
        radius[i] = randFloat(0.5, 0.5);
        idxx[i] = i;
        idxy[i] = i;
        idxz[i] = i;
        
    }*/

    int cell_count = 2;//floorf(n_objects / particles_per_cell);
    thrust::host_vector<int> cells(cell_count * cell_count *  particles_per_cell);
    thrust::host_vector<int> temp_cells(cell_count * cell_count * particles_per_cell);
    thrust::host_vector<int> cell_occupation(cell_count * cell_count);
    thrust::host_vector<int> cell_prefix(cell_count * cell_count);
    thrust::host_vector<int> obj_to_cells(cell_count * cell_count * particles_per_cell);

    

    position[0] = make_float4(1.0, 0.5, 0.5, 0.0);
    position[1] = make_float4(0.0, 0.25, 0.25, 0.0);
    //position[2] = make_float4(0.0, 0.0, 1.0, 0.0);
    radius[0] = 0.2;
    radius[1] = 0.2;
    //radius[2] = 0.5;
    idxx[0] = 0;
    idxx[1] = 1;
    //idxz[0] = 0;
    //idxx[1] = 1;
    //idxy[1] = 1;
    //idxz[1] = 1;
    //idxx[2] = 2;
    //idxy[2] = 2;
    //idxz[2] = 2;

    for (int i = 0; i < max_collisions * n_objects + max_collisions; i++) {
        potential_collision[i] = -1;
    }

    for (int i = 0; i < cell_count * cell_count * particles_per_cell; i++) {
        cells[i] = -1;
        temp_cells[i] = -1;
    }

    computeAABB(position, radius, lower_bound, upper_bound);

    //// Copy host_vector H to device_vector D
    thrust::device_vector<float4> d_position = position;
    thrust::device_vector<float> d_radius = radius;
    thrust::device_vector<float4> d_lower_bound = lower_bound;
    thrust::device_vector<float4> d_upper_bound = upper_bound;

    thrust::device_vector<float> d_lowerx = lowerx;
    thrust::device_vector<float> d_upperx = upperx;
    thrust::device_vector<float> d_lowery = lowery;
    thrust::device_vector<float> d_uppery = uppery;
    thrust::device_vector<float> d_lowerz = lowerz;
    thrust::device_vector<float> d_upperz = upperz;

    thrust::device_vector<int> d_idxx = idxx;
    thrust::device_vector<int> d_idxy = idxy;
    thrust::device_vector<int> d_idxz = idxz;

    thrust::device_vector<int> d_potential_collision = potential_collision;

    thrust::device_vector<int> d_sorted_idxx = sorted_idxx;
    thrust::device_vector<float> d_sorted_lowerx = sorted_lowerx;

    thrust::device_vector<int> d_radix = radix;
    thrust::device_vector<int> d_radix_sum = radix_sum;

    thrust::device_vector<int> d_temp_cells = temp_cells;
    thrust::device_vector<int> d_cells = cells;
    thrust::device_vector<int> d_cell_occ = cell_occupation;
    thrust::device_vector<int> d_cell_prefix = cell_prefix;

    thrust::device_vector<int> d_toic = total_objs_in_cells;
    thrust::device_vector<int> d_obj_to_cells = obj_to_cells;

    // First cast all device_vectors to pointers
    float4* d_position_ptr = thrust::raw_pointer_cast(d_position.data());
    float4* d_lower_bound_ptr = thrust::raw_pointer_cast(d_lower_bound.data());
    float4* d_upper_bound_ptr = thrust::raw_pointer_cast(d_upper_bound.data());
    float* d_radius_ptr = thrust::raw_pointer_cast(d_radius.data());

    float* d_upperx_ptr = thrust::raw_pointer_cast(d_upperx.data());
    float* d_lowerx_ptr = thrust::raw_pointer_cast(d_lowerx.data());
    float* d_uppery_ptr = thrust::raw_pointer_cast(d_uppery.data());
    float* d_lowery_ptr = thrust::raw_pointer_cast(d_lowery.data());
    float* d_upperz_ptr = thrust::raw_pointer_cast(d_upperz.data());
    float* d_lowerz_ptr = thrust::raw_pointer_cast(d_lowerz.data());

    int* d_idxx_ptr = thrust::raw_pointer_cast(d_idxx.data());
    int* d_idxy_ptr = thrust::raw_pointer_cast(d_idxy.data());
    int* d_idxz_ptr = thrust::raw_pointer_cast(d_idxz.data());

    int* d_potential_collision_ptr = thrust::raw_pointer_cast(d_potential_collision.data());

    float* d_sorted_lowerx_ptr = thrust::raw_pointer_cast(d_sorted_lowerx.data());
    int* d_sorted_idxx_ptr = thrust::raw_pointer_cast(d_sorted_idxx.data());

    int* d_radix_ptr = thrust::raw_pointer_cast(d_radix.data());
    int* d_radix_sum_ptr = thrust::raw_pointer_cast(d_radix_sum.data());

    int* d_temp_cells_ptr = thrust::raw_pointer_cast(d_temp_cells.data());
    int* d_cells_ptr = thrust::raw_pointer_cast(d_cells.data());
    int* d_cells_occ_ptr = thrust::raw_pointer_cast(d_cell_occ.data());
    int* d_cell_prefix_ptr = thrust::raw_pointer_cast(d_cell_prefix.data());

    int* d_toic_ptr = thrust::raw_pointer_cast(d_toic.data());
    int* d_obj_to_cells_ptr = thrust::raw_pointer_cast(d_obj_to_cells.data());
    

    int threadsPerBlock = 256;
    int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

    std::chrono::time_point<std::chrono::system_clock> start1;
    std::chrono::duration<double> duration1;

    double time1;
    start1 = std::chrono::system_clock::now();

    // Map objects into cells
    mapObjectsToCells << < numBlocks, threadsPerBlock >> > (
        n_objects, 
        d_idxx_ptr, 
        d_lower_bound_ptr, 
        d_upper_bound_ptr, 
        global_lower, 
        global_upper, 
        cell_count, 
        d_temp_cells_ptr,
        particles_per_cell,
        d_cells_occ_ptr,
        d_cell_prefix_ptr
        );

    int threadsPerBlockCells = 256;
    int numBlocksCells = ((cell_count * cell_count * particles_per_cell) + threadsPerBlock - 1) / threadsPerBlock;

    launchPrefixScan(
        d_cell_prefix_ptr,
        d_cells_occ_ptr,
        cell_count * cell_count);

    int toic = thrust::reduce(thrust::device, d_cells_occ_ptr, d_cells_occ_ptr + (cell_count * cell_count));

    reduceCellArray << < numBlocksCells, threadsPerBlockCells >> > (
        cell_count* cell_count,
        cell_count* cell_count* particles_per_cell,
        d_cell_prefix_ptr,
        d_cells_occ_ptr,
        d_temp_cells_ptr,
        d_cells_ptr,
        particles_per_cell,
        d_obj_to_cells_ptr
        );
    
    // Project to x-axis, make this
    projectAxis << < numBlocks, threadsPerBlock >> > (
        toic,
        d_obj_to_cells_ptr,
        d_cells_ptr,
        d_cells_occ_ptr,
        d_cell_prefix_ptr,
        global_lower,
        global_upper,
        d_lower_bound_ptr,
        d_upper_bound_ptr,
        d_lowerx_ptr,
        d_upperx_ptr,
        d_lowery_ptr,
        d_uppery_ptr,
        d_lowerz_ptr,
        d_upperz_ptr,
        n_objects
        ); 

    radixSortCustom(
        (uint32_t*)d_lowerx_ptr,
        (uint32_t*)d_idxx_ptr,
        (uint32_t*)d_sorted_lowerx_ptr,
        (uint32_t*)d_sorted_idxx_ptr,
        (uint32_t*)d_radix_ptr,
        (uint32_t*)d_radix_sum_ptr,
        n_objects
    );


    //std::cout << "Launching kernel\n";
    sweep << <numBlocks, threadsPerBlock >> > (
        d_upperx_ptr,
        d_sorted_lowerx_ptr,
        d_uppery_ptr,
        d_lowery_ptr,
        d_upperz_ptr,
        d_lowerz_ptr,
        d_sorted_idxx_ptr,
        d_idxy_ptr,
        d_idxz_ptr,
        d_potential_collision_ptr,
        n_objects,
        max_collisions);

    /*sweepBlocks << <n_objects, threadsPerBlock >> > (
        d_upperx_ptr,
        d_lowerx_ptr,
        d_uppery_ptr,
        d_lowery_ptr,
        d_upperz_ptr,
        d_lowerz_ptr,
        d_idxx_ptr,
        d_idxy_ptr,
        d_idxz_ptr,
        d_potential_collision_ptr,
        n_objects,
        max_collisions);*/

    //std::cout << "Kernel completed\n";

    duration1 = std::chrono::system_clock::now() - start1;

    time1 = duration1.count();

   
    CUDA_ERR_CHECK(cudaDeviceSynchronize());

    int sum = thrust::count_if(thrust::device,  d_potential_collision.begin(), d_potential_collision.end(), isValid());

    //int sum = thrust::reduce(thrust::device, d_potential_collision.begin(), d_potential_collision.end(), 0, thrust::plus<int>());
    std::cout << sum << " Collisions detected in " << time1 << " secs\n" << std::endl;

    std::cout << "Number of objects (AABBs): " << n_objects << "\n";

    cells = d_cells;
    cell_prefix = d_cell_prefix;
    cell_occupation = d_cell_occ;
    total_objs_in_cells = d_toic;

    printf("Total possible collisions: %d\n", toic);
    
    printf("Cell [ ");
        for (int j = 0; j < particles_per_cell * cell_count * cell_count; j++) {
            printf("%d, ", cells[j]);
        }
    printf(" ]\n");

    printf("Cell occupation [ ");
    for (int j = 0; j < cell_count * cell_count; j++) {
        printf("%d, ", cell_occupation[j]);
    }
    printf(" ]\n");


    return 0;
}