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

	typedef struct Sphere {
		unsigned int clump_owner; // Allows owner to be accessed

		float4 relative_position; // Sphere has no worldspace position, it only exists in reference to the clump

		float4 force;

		float mass;
		float radius;

		nvstd::function<void()> support_function_ptr; // Assigned at runtime on the device

	} Sphere;

	typedef struct Triangle {
		unsigned int mesh_owner; // Allows owner to be accessed

		// Vertex data
		unsigned int vertex_idx;
		unsigned int n_vertices = 3;

		// Simulation data
		float4 relative_position;
		float4 quaternion;

		float4 force;
		float4 torque;

		nvstd::function<void()> support_function_ptr; // Assigned at runtime on the device

	} Triangle;

	typedef struct Polyhedra {
		// Master data
		bool is_active;
		bool is_visible;

		// Vertex data
		unsigned int vertex_idx;
		unsigned int n_vertices;

		// Simulation data
		float mass;

		float4 position;
		float4 quaternion = make_float4(1.0f, 0.0f, 0.0f, 0.0f);

		float4 linear_velocity;
		float4 angular_velocity;

		float4 linear_momentum;
		float4 angular_momentum;

		float4 force;
		float4 torque;

		float9 inertial_tensor;

		nvstd::function<void()> support_function_ptr; // Assigned at runtime on the device

	} Polyhedra;

	/// <summary>
	/// Unified Primitive Representation - Convex Hull
	/// - All derived Groups are collections of one or more primitives
	/// </summary>

	typedef struct Hull {
		unsigned int type; // sphere: 0, tri: 1, polyhedra: 2

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
