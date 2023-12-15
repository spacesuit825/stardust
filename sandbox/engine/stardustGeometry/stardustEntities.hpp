#ifndef _STARDUST_ENTITY_HEADER_
#define _STARDUST_ENTITY_HEADER_

// CUDA
#include <cuda_runtime.h>

// Internal
#include "../stardustUtility/helper_math.hpp"
#include "../stardustUtility/types.hpp"

#define CLUMP 0
#define MESH 1
#define COMPLEX_POLYHEDRON 2

namespace STARDUST {

	typedef struct Entity {
		int type;

		bool is_active;
		bool is_visible;

		// If we need to access primitives we can
		int primitive_idx;
		int n_primitives; 

		float mass;

		// Simulation data
		float4 position; // COM of the entity

		float4 velocity;
		float4 angular_velocity;

		float4 linear_momentum;
		float4 angular_momentum;

		float4 quaternion;

		float9 inertia_tensor;

		float4 force;
		float4 torque;

	};
}

#endif // _STARDUST_ENTITY_HEADER_