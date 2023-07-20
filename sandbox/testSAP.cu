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
#include <cub/device/device_radix_sort.cuh>

#include <../src/engine/cuda/collision_detection.cuh>
#include <../src/engine/cuda/cuda_utils.hpp>

#include <iostream>

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

    int n_tid = tid + 10;
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

    if (tid >= n_objects) {
        return;
    }

    float4 lower_bound = d_lower_bound_ptr[tid];
    float4 upper_bound = d_upper_bound_ptr[tid];

    d_lowerx_ptr[tid] = lower_bound.x;
    d_lowery_ptr[tid] = lower_bound.y;
    d_lowerz_ptr[tid] = lower_bound.z;

    d_upperx_ptr[tid] = upper_bound.x;
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

int main() {

    int n_objects = 200000;
    int max_collisions = 10;

     //H has storage for 4 integers
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

    for (int i = 0; i < n_objects; i++) {
        position[i] = make_float4(randFloat(0.5, 100.0), randFloat(0.5, 100.0), randFloat(0.5, 100.0), 0.0f);
        radius[i] = randFloat(0.5, 0.5);
        idxx[i] = i;
        idxy[i] = i;
        idxz[i] = i;
        
    }

    /*position[0] = make_float4(0.0, 0.0, 0.0, 0.0);
    position[1] = make_float4(0.0, 0.0, 0.5, 0.0);
    position[2] = make_float4(0.0, 0.0, 1.0, 0.0);
    radius[0] = 0.5;
    radius[1] = 0.5;
    radius[2] = 0.5;
    idxx[0] = 0;
    idxy[0] = 0;
    idxz[0] = 0;
    idxx[1] = 1;
    idxy[1] = 1;
    idxz[1] = 1;
    idxx[2] = 2;
    idxy[2] = 2;
    idxz[2] = 2;*/

    for (int i = 0; i < max_collisions * n_objects + max_collisions; i++) {
        potential_collision[i] = -1;
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

    int threadsPerBlock = 256;
    int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;

    std::chrono::time_point<std::chrono::system_clock> start1;
    std::chrono::duration<double> duration1;

    double time1;
    start1 = std::chrono::system_clock::now();

    // Project to x-axis, make this
    projectAxis << < numBlocks, threadsPerBlock >> > (
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

    
    /*projectAABBx(d_lower_bound, d_upper_bound, d_lowerx, d_upperx);
    projectAABBy(d_lower_bound, d_upper_bound, d_lowery, d_uppery);
    projectAABBz(d_lower_bound, d_upper_bound, d_lowerz, d_upperz);*/

    

    radixSortCustom(
        (uint32_t*)d_lowerx_ptr,
        (uint32_t*)d_idxx_ptr,
        (uint32_t*)d_sorted_lowerx_ptr,
        (uint32_t*)d_sorted_idxx_ptr,
        (uint32_t*)d_radix_ptr,
        (uint32_t*)d_radix_sum_ptr,
        n_objects
    );

    //CUDA_ERR_CHECK(cudaDeviceSynchronize());

    /*d_lowerx_ptr = (float*)d_lowerx_ptr;
    d_idxx_ptr = (int*)d_idxx_ptr;
    d_sorted_lowerx_ptr = (float*)d_sorted_lowerx_ptr;
    d_sorted_idxx_ptr = (int*)d_sorted_idxx_ptr;
    d_radix_ptr = (int*)d_radix_ptr;
    d_radix_sum_ptr = (int*)d_radix_sum_ptr;*/

    //radixSort(d_lowerx_ptr, d_idxx_ptr, n_objects);
    //// Radix sort
    //// Determine temporary device storage requirements
    //void* d_temp_storage = NULL;
    //size_t temp_storage_bytes = 0;

    //cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
    //    d_lowerx_ptr, d_sorted_lowerx_ptr, d_idxx_ptr, d_sorted_idxx_ptr, n_objects);

    //// Allocate temporary storage
    //cudaMalloc(&d_temp_storage, temp_storage_bytes);

    //// Run sorting operation
    //cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
    //    d_lowerx_ptr, d_sorted_lowerx_ptr, d_idxx_ptr, d_sorted_idxx_ptr, n_objects);

    //cudaFree(d_temp_storage);

   

    //
    //radixSort(d_lowery, d_idxy);
    //radixSort(d_lowerz, d_idxz);

    // Perform the sweep
   /* int threadsPerBlock = 256;
    int numBlocks = (n_objects + threadsPerBlock - 1) / threadsPerBlock;*/

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

   
    int sum = thrust::count_if(thrust::device,  d_potential_collision.begin(), d_potential_collision.end(), isValid());

    //int sum = thrust::reduce(thrust::device, d_potential_collision.begin(), d_potential_collision.end(), 0, thrust::plus<int>());
    std::cout << sum << " Collisions detected in " << time1 << " secs\n" << std::endl;

    std::cout << "Number of objects (AABBs): " << n_objects << "\n";
    



    return 0;
}