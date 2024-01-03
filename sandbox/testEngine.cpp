#include "stardustCollision/stardustBroadPhase/stardustLBVH.hpp"
#include "stardustCollision/stardustNarrowPhase/stardustMPR.hpp"
#include "stardustGeometry/stardustEntityHandler.hpp"
#include "stardustDynamics/stardustEngine.hpp"
#include "stardustGeometry/stardustPrimitives.hpp"
#include "stardustUtility/load_data.hpp"
// #include "stardustUtility/cuda_utils.cuh"

#include <iostream>





int main() {

	STARDUST::EngineParameters engine_parameters;
	engine_parameters.max_broad_collisions = 10000;
	engine_parameters.max_narrow_collisions = 1000;
	engine_parameters.time_step = 1e-4f;
	
	STARDUST::LBVH lbvh;
	STARDUST::MPR mpr;

	STARDUST::EntityHandler entity_handler;

	STARDUST::Engine engine(engine_parameters);

	float4 vertex1 = make_float4(-1.0f, -1.0f, 0.0f, 0.0f);
	float4 vertex2 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
	float4 vertex3 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
	float4 vertex4 = make_float4(-1.0f, -1.0f, 0.0f, 0.0f);

	std::vector<float4> vertices;
	std::vector<int> indices;

	STARDUST::Triangle triangle;
	vertices.push_back(vertex1);
	vertices.push_back(vertex2);
	vertices.push_back(vertex3);
	//vertices.push_back(vertex4);

	indices = {0, 1, 2};
	
	triangle.position = 1 / 3.0f * (vertex1 + vertex2 + vertex3);
	triangle.mass = 0.5f;
	triangle.normal_stiffness = 1e+08;
	triangle.damping = 20.0f;
	triangle.tangential_stiffness = 10.0f;

	triangle.vertices = vertices;
	triangle.indices = indices;

	STARDUST::Sphere sphere2;
	sphere2.position = make_float4(-1.0f, -1.0f, -2.0f, 0.0f);
	sphere2.radius = 0.5f;
	sphere2.mass = 10000000.0f;
	sphere2.normal_stiffness = 1e+04;
	sphere2.damping = 0.02f;
	sphere2.tangential_stiffness = 1e+02;

	STARDUST::Sphere sphere3;
	sphere3.position = make_float4(0.5f, 0.3f, 1.0f, 0.0f);
	sphere3.radius = 0.5f;
	sphere3.mass = 2.0f;
	sphere3.normal_stiffness = 1e+07;
	sphere3.damping = 2.0f;
	sphere3.tangential_stiffness = 1e+02;





	STARDUST::Sphere sphere3_5;
	sphere3_5.position = make_float4(0.0f, 0.27f, 5.3f, 0.0f);
	sphere3_5.radius = 0.5f;
	sphere3_5.mass = 5.0f;
	sphere3_5.normal_stiffness = 1e+07;
	sphere3_5.damping = 20.0f;
	sphere3_5.tangential_stiffness = 0.05f;


	STARDUST::Sphere sphere4;
	sphere4.position = make_float4(0.0f, 0.0f, 1.3f, 0.0f);
	sphere4.radius = 0.5f;
	sphere4.mass = 3.7f;
	sphere4.normal_stiffness = 1e+07;
	sphere4.damping = 20.0f;
	sphere4.tangential_stiffness = 10.0f;

	STARDUST::Sphere sphere5;
	sphere5.position = make_float4(0.0f, 0.3f, 2.5f, 0.0f);
	sphere5.radius = 0.5f;
	sphere5.mass = 3.7f;
	sphere5.normal_stiffness = 1e+07;
	sphere5.damping = 20.0f;
	sphere5.tangential_stiffness = 0.05f;

	STARDUST::Sphere sphere6;
	sphere6.position = make_float4(0.0f, 0.32f, 6.7f, 0.0f);
	sphere6.radius = 0.5f;
	sphere6.mass = 3.7f;
	sphere6.normal_stiffness = 1e+07;
	sphere6.damping = 20.0f;
	sphere6.tangential_stiffness = 0.05f;

	std::vector<STARDUST::Polyhedron> polyhedrons;
	std::vector<float4> poly_vert;
	std::vector<int> poly_idx;

	STARDUST::Polyhedron polyhedron;

	float4 p1 = make_float4(-0.5f, -0.5f, 1.0f, 0.0f);
	float4 p2 = make_float4(0.5f, 0.0f, 0.5f, 0.0f);
	float4 p3 = make_float4(0.0f, 0.5f, 0.5f, 0.0f);
	float4 p4 = make_float4(-0.5f, -0.5f, 0.5f, 0.0f);

	poly_idx = { 0, 1, 3, 1, 2, 3, 0, 2, 3, 0, 2, 1 };

	poly_vert.push_back(p1);
	poly_vert.push_back(p2);
	poly_vert.push_back(p3);
	poly_vert.push_back(p4);

	polyhedron.position = (1 / 4.0f) * (p1 + p2 + p3 + p4);
	// polyhedron.position = make_float4(-0.125f, -0.125f, 0.625f, 0.0f);
	polyhedron.mass = 0.5f;
	polyhedron.normal_stiffness = 1e+07;
	polyhedron.damping = 20.0f;
	polyhedron.tangential_stiffness = 10.0f;

	polyhedron.vertices = poly_vert;
	polyhedron.indices = poly_idx;

	std::cout << polyhedron.position.z << std::endl;

	std::vector<STARDUST::Triangle> triangles;
	// complex_polyhedron.push_back(polyhedron);

	std::vector<STARDUST::Sphere> clump2;
	std::vector<STARDUST::Sphere> clump3;
	std::vector<STARDUST::Sphere> clump4;
	std::vector<STARDUST::Sphere> clump5;
	std::vector<STARDUST::Sphere> clump6;
	
	clump2.push_back(sphere2);
	clump3.push_back(sphere3_5);
	
	clump5.push_back(sphere5);
	clump6.push_back(sphere6);
	clump4.push_back(sphere4);

	triangles.push_back(triangle);
	polyhedrons.push_back(polyhedron);

	
	
	//load_stl("C:/Users/lachl/Documents/stardust/sandbox/test_stl.stl", triangles);

	entity_handler.addEntity(clump4);

	entity_handler.addEntity(triangles);

	entity_handler.addEntity(polyhedrons);

	

	std::cout << entity_handler.getNPrimitives() << std::endl;
	
	
	engine.setupEngine(
		lbvh,
		mpr,
		entity_handler
	);

	engine.allocate();

	for (int i = 0; i < 8000; i++) {
		//std::cout << "-------------------------------------------\n";
		engine.run();

		if (i % 80 == 0) {
			engine.writeToVTK(i);
		}

		engine.reset();
	}

	//loadPrimitiveData("C:/Users/lachl/Documents/stardust/sandbox/particles.txt");
}