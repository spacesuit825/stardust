#include <vector_types.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <device_functions.h>

struct Node
{
	uint32_t parent_idx;
	uint32_t left_idx;
	uint32_t right_idx;
	uint32_t object_idx; // == 0xFFFFFFFF if internal node.
};

struct AABB
{
	float4 upper_extent;
	float4 lower_extent;

	// initialize empty box
	inline __host__ __device__
		AABB() : upper_extent(make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f)), lower_extent(make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.0f))
	{}

	// initialize a box containing a single point
	inline __host__ __device__
		AABB(const float4& p, float radius) : upper_extent(make_float4(p.x + radius, p.y + radius, p.z + radius, 0.0f)), lower_extent(make_float4(p.x - radius, p.y - radius, p.z - radius, 0.0f))
	{}
};

//__host__ __device__ int tagPoints(
//	const float4& p,
//	AABB aabb,
//	int max_level
//)
//{
//	int tag = 0;
//
//	for (int level = 1; level <= max_level; ++level) {
//
//		// Classify in x-direction
//		float ymid = 0.5f * (aabb.ymin + aabb.ymax);
//		int y_half = (p.x < ymid) ? 0 : 1;
//
//		tag |= y_half;
//		tag <<= 1;
//
//		// Classify in y-direction
//		float zmid = 0.5f * (aabb.zmin + aabb.zmax);
//		int z_half = (p.z < ymid) ? 0 : 1;
//
//		tag |= z_half;
//		tag <<= 1;
//
//		aabb.ymin = (y_half) ? ymid : aabb.ymin;
//		aabb.ymax = (y_half) ? aabb.ymax : ymid;
//		aabb.zmin = (z_half) ? zmid : aabb.zmin;
//		aabb.zmax = (z_half) ? aabb.zmax : zmid;
//	}
//
//	tag >>= 1;
//	
//	return tag;
//}

//struct ClassifyPoint {
//	AABB aabb;
//	int max_level;
//
//	ClassifyPoint(const AABB bounds, int max_level) :
//		aabb(bounds),
//		max_level(max_level)
//	{}
//
//	inline __device__ __host__
//		int operator()(const float4& p)
//	{
//		return tagPoints(p, aabb, max_level);
//	}
//};

