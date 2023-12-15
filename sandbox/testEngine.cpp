#include "stardustCollision/stardustBroadPhase/stardustLBVH.hpp"
#include "stardustCollision/stardustNarrowPhase/stardustMPR.hpp"
#include "stardustGeometry/stardustEntityHandler.hpp"
#include "stardustDynamics/stardustEngine.hpp"
#include "stardustGeometry/stardustPrimitives.hpp"
//#include "stardustUtility/cuda_utils.cuh"

#include <iostream>





int main() {

	STARDUST::EngineParameters engine_parameters;
	engine_parameters.max_broad_collisions = 10000;
	engine_parameters.max_narrow_collisions = 1000;
	engine_parameters.time_step = 1e-3f;
	
	STARDUST::LBVH lbvh;
	STARDUST::MPR mpr;

	STARDUST::EntityHandler entity_handler;

	STARDUST::Engine engine(engine_parameters);

	STARDUST::Sphere sphere1;
	sphere1.position = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	sphere1.radius = 0.5f;
	sphere1.mass = 1.0f;
	sphere1.normal_stiffness = 1e+07;
	sphere1.damping = 0.02f;
	sphere1.tangential_stiffness = 1e+02;

	STARDUST::Sphere sphere2;
	sphere2.position = make_float4(0.5f, 0.0f, 0.0f, 0.0f);
	sphere2.radius = 0.5f;
	sphere2.mass = 1.0f;
	sphere2.normal_stiffness = 1e+07;
	sphere2.damping = 0.02f;
	sphere2.tangential_stiffness = 1e+02;

	std::vector<STARDUST::Sphere> clump1;
	clump1.push_back(sphere1);

	std::vector<STARDUST::Sphere> clump2;
	clump2.push_back(sphere2);

	entity_handler.addEntity(clump1);
	entity_handler.addEntity(clump2);
	
	engine.setupEngine(
		lbvh,
		mpr,
		entity_handler
	);

	engine.allocate();

	for (int i = 0; i < 10; i++) {
		engine.run();
	}
}