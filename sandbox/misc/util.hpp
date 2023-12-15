#ifndef _STARDUST_UTILITY_HEADER_
#define _STARDUST_UTILITY_HEADER_

#include <vector_types.h>
#include <cuda_runtime.h>
#include <cstdint>
//#include <device_functions.h>



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

#endif // _STARDUST_UTILITY_HEADER_
