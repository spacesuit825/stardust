#ifndef _STARDUST_ENTITY_HEADER_
#define _STARDUST_ENTITY_HEADER_

#include "types.hpp"
#include <vector>
#include "helper_math.hpp"

namespace STARDUST {
	
	struct DEMSphere {
		// DEM Particle
		float4 position;
		Scalar size;
		Scalar mass;
	};

	class DEMParticle {
		// DEM Entity
		//	- Conglomerate of DEM particles
		//  - For now you can only load cubes (default is side length of 0.01m or 1 cm)
		// TODO: add mesh loading for arbitrary entity shapes
	public:
		DEMParticle(unsigned int id, int grid_resolution, Scalar size, Scalar diameter, Scalar mass, float4 position, float4 velocity)
			: m_id(id), m_grid_resolution(grid_resolution), m_size(size), diameter(diameter), mass(mass), position(position), velocity(velocity) {
			
			COM = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			initParticles(m_grid_resolution); // Create particles in Entity Space
			getCenterOfMass(); // Compute Center of Mass of the entity
			setParticlesInWorldSpace(); // Move the particles into the World Space

		}

		void initParticles(int);
		void setParticlesInWorldSpace();
		// computeBoundingBox(mesh); // <--- Add this for generating bounds on the mesh when mesh loading is added
		void getCenterOfMass();
		

		std::vector<DEMSphere> getParticles() { return particles; };
		
		Scalar* convertParticlePositionsToNakedArray(); // Mostly for testing

		float4 position; // Position of entity in World Space (at its upper left bounding box corner)
		float4 velocity;
		float4 COM;
		float mass;
		float diameter;

	private:
		unsigned int m_id;

		int m_n_particles;
		

		Scalar m_size;

		int m_grid_resolution; // The mesh is divided into particles based on this resolution

		std::vector<DEMSphere> particles;

	};

	class DEMMesh {
		// Lightweight class to generate a mesh struct for the GPU

	//public:
	//	//DEMMesh(float4* vertices, int* indicies, int n_triangles);

	//	int n_triangles;
	//	float4 position;
	//	float4 quat;

	//	std::vector<float4> vertices;
	//	std::vector<int> indicies;

	//private:
		


	};

}

#endif // _STARDUST_ENTITY_HEADER_