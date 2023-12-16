#ifndef _STARDUST_ENTITY_HANDLER_
#define _STARDUST_ENTITY_HANDLER_

// C++
#include <vector>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Internal
#include "stardustEntities.hpp"
#include "stardustPrimitives.hpp"
#include "../stardustUtility/helper_math.hpp"
#include "../stardustUtility/util.hpp"



// The entity handler is responsible for coordinating, storing, allocating and destroying entities on the host and device

namespace STARDUST {

	struct DeviceGeometryData {
		Hull* d_hull_ptr;
		Entity* d_entity_ptr;
		float4* d_init_vertex_ptr;
		float4* d_vertex_ptr;
		AABB* d_aabb_ptr;
	};

	class EntityHandler {

	public:

		void addEntity(std::vector<Sphere> clump);
		void addEntity(std::vector<Triangle> mesh);
		void addEntity(std::vector<Polyhedron> complex_polyhedron);
		//void removeEntity();

		AABB computeBoundingBox(Sphere& sphere);
		AABB computeBoundingBox(Triangle& triangle);
		AABB computeBoundingBox(Polyhedron& polyhedron);

		void allocate();
		void writeToVTK(int time_step);
		
		DeviceGeometryData& getDeviceData() { return device_geometry_data; };
		
		unsigned int getNPrimitives() { return hulls.size(); };
		unsigned int getNEntities() { return entities.size(); };


	private:


		float padding = 0.05f;

		int n_primitives;
		int n_entities;

		DeviceGeometryData device_geometry_data;


		thrust::host_vector<Hull> hulls;
		thrust::host_vector<Entity> entities;
		thrust::host_vector<float4> init_vertex;
		thrust::host_vector<float4> vertex;
		thrust::host_vector<AABB> aabb;

		thrust::device_vector<Hull> d_hull;
		thrust::device_vector<Entity> d_entity;
		thrust::device_vector<float4> d_init_vertex;
		thrust::device_vector<float4> d_vertex;
		thrust::device_vector<AABB> d_aabb;

		Hull* d_hull_ptr;
		Entity* d_entity_ptr;
		float4* d_init_vertex_ptr;
		float4* d_vertex_ptr;
		AABB* d_aabb_ptr;

	};


}





#endif // _STARDUST_ENTITY_HANDLER_