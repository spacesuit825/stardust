#ifndef _STARDUST_PRIMITIVE_HEADER_
#define _STARDUST_PRIMITIVE_HEADER_

// CUDA
#include <cuda_runtime.h>

// Internal
#include "helper_math.hpp"
#include "helper_math.hpp"
#include "types.hpp"

#define SPHERE 0
#define TRIANGLE 1
#define POLYHEDRA 2

// Key simulation types
// - primitives
//	 1. sphere
//   2. triangle

// - entities
//	 1. clump - one or more spheres
//   2. world mesh - triangle soup
//   3. polyhedra - convex hull

// [Clump ... Clump]
// [Sphere, sphere ... sphere, sphere]
// [0, 0 ... 1, 1]

namespace STARDUST {

	/// <summary>
	/// Unified Primitive Representation - Convex Hull
	/// - All derived Groups are collections of one or more primitives
	/// </summary>

	typedef struct Hull {
		unsigned int type; // sphere: 0, tri: 1, polyhedra: 2

		unsigned int group_owner; // Group to which this primitve belongs

		bool is_active;
		bool is_visible;

		// Vertex data
		int vertex_idx; // For a sphere this is -1
		int n_vertices; // For triangle this is 3 and for sphere this is 0

		float mass; // Triangles are given a default mass of 1kg, but are non-dynamic by their nature
		float radius; // For all primitives except sphere this is assigned to -1.0

		// Simulation data
		float4 position; // Centroid/Barycentre for all primitives
		float4 force;
		float4 torque;

	};
}


#endif // !_STARDUST_PRIMITIVE_HEADER_
