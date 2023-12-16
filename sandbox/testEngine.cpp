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

	float4 vertex1 = make_float4(-1.0f, -1.0f, 1.0f, 0.0f);
	float4 vertex2 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
	float4 vertex3 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
	float4 vertex4 = make_float4(-1.0f, -1.0f, 0.0f, 0.0f);

	std::vector<float4> vertices;
	std::vector<int> indices;

	STARDUST::Polyhedron polyhedron;
	vertices.push_back(vertex1);
	vertices.push_back(vertex2);
	vertices.push_back(vertex3);
	vertices.push_back(vertex4);

	indices = { 0, 3, 1, 0, 1, 2, 2, 3, 0, 1, 2, 3 };
	
	polyhedron.position = 1 / 3 * (vertex1 + vertex2 + vertex3 + vertex4);
	polyhedron.mass = 1.0f;
	polyhedron.normal_stiffness = 1e+05;
	polyhedron.damping = 0.02f;
	polyhedron.tangential_stiffness = 1e+02;

	polyhedron.vertices = vertices;
	polyhedron.indices = indices;

	STARDUST::Sphere sphere2;
	sphere2.position = make_float4(-1.0f, -1.0f, -2.0f, 0.0f);
	sphere2.radius = 0.5f;
	sphere2.mass = 10000000.0f;
	sphere2.normal_stiffness = 1e+04;
	sphere2.damping = 0.02f;
	sphere2.tangential_stiffness = 1e+02;

	STARDUST::Sphere sphere3;
	sphere3.position = make_float4(0.0f, 0.2f, 2.5f, 0.0f);
	sphere3.radius = 0.5f;
	sphere3.mass = 2.0f;
	sphere3.normal_stiffness = 1e+04;
	sphere3.damping = 0.02f;
	sphere3.tangential_stiffness = 1e+02;

	STARDUST::Sphere sphere4;
	sphere4.position = make_float4(0.3f, 0.7f, 2.0f, 0.0f);
	sphere4.radius = 0.5f;
	sphere4.mass = 2.0f;
	sphere4.normal_stiffness = 1e+04;
	sphere4.damping = 0.02f;
	sphere4.tangential_stiffness = 1e+02;

	std::vector<STARDUST::Polyhedron> complex_polyhedron;
	complex_polyhedron.push_back(polyhedron);

	std::vector<STARDUST::Sphere> clump2;
	std::vector<STARDUST::Sphere> clump3;
	std::vector<STARDUST::Sphere> clump4;

	clump2.push_back(sphere2);
	clump3.push_back(sphere3);
	clump4.push_back(sphere4);

	entity_handler.addEntity(clump2);
	entity_handler.addEntity(complex_polyhedron);
	
	//entity_handler.addEntity(clump3);
	entity_handler.addEntity(clump4);
	//entity_handler.addEntity(clump5);
	
	engine.setupEngine(
		lbvh,
		mpr,
		entity_handler
	);

	engine.allocate();

	for (int i = 0; i < 3000; i++) {
		std::cout << "-------------------------------------------\n";
		engine.run();
		engine.writeToVTK(i);
		engine.reset();
	}
}