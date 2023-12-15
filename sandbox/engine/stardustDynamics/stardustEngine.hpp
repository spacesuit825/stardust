#ifndef _STARDUST_DYNAMICS_INTEGRATION_HEADER_
#define _STARDUST_DYNAMICS_INTEGRATION_HEADER_

#include "../stardustCollision/stardustBroadPhase/stardustLBVH.hpp"
#include "../stardustCollision/stardustNarrowPhase/stardustMPR.hpp"
#include "../stardustGeometry/stardustEntityHandler.hpp"


namespace STARDUST {

	enum IntergrationMethod {
		Euler,
		Verlet,
		Jolt
	};

	enum BroadPhaseCollisionMethod {
		//LBVH,
		UniformGrid,
		SAP
	};

	enum NarrowPhaseCollisionMethod {
		GJK_EPA,
		//MPR,
		SAT
	};

	typedef struct EngineParameters {
		IntergrationMethod integration_method;
		BroadPhaseCollisionMethod broad_method;
		NarrowPhaseCollisionMethod narrow_method;

		int max_broad_collisions;
		int max_narrow_collisions;

		double time_step;

	} EngineParameters;


	class Engine {
		
	public:

		Engine(EngineParameters engine_params) : engine_parameters(engine_params) {
		
		}

		~Engine() {

		}

		void setupEngine(LBVH lbvh, MPR mpr, EntityHandler entity_handler);

		void allocate();

		void run();

		//void destroy();

		
	private:

		void updatePrimitives(int n_primitives, Hull* d_hull_ptr, Entity* d_entity_ptr, AABB* d_aabb_ptr);
		void resolveCollisions(int n_collisions, CollisionManifold* d_collision_manifold_ptr, Hull* d_hull_ptr);
		void collateForcesAndTorques(int n_entities, Hull* d_hull_ptr, Entity* d_entity_ptr);
		void integrateForward(int n_entities, float time_step, Entity* d_entity_ptr);

		EngineParameters engine_parameters;

		LBVH lbvh;
		MPR mpr;

		EntityHandler entity_handler;

	};
}



#endif // _STARDUST_DYNAMICS_INTEGRATION_HEADER_