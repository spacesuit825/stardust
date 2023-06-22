#ifndef _STARDUST_ENTITY_HEADER_
#define _STARDUST_ENTITY_HEADER_

#include "types.hpp"
#include <vector>

namespace STARDUST {
	
	struct DEMParticle {
	// DEM Particle
		Vec3f position; // Position of particle in Entity Space
		Scalar density;
		Scalar size;
	};

	class DEMEntity {
		// DEM Entity
		//	- Conglomerate of DEM particles
		//  - For now you can only load cubes (default is side length of 0.01m or 1 cm)
		// TODO: add mesh loading for arbitrary entity shapes
	public:
		DEMEntity(unsigned int id, int grid_resolution, Scalar size, Vec3f position, Vec3f velocity) 
			: m_id(id), m_grid_resolution(grid_resolution), m_size(size), m_position(position), m_velocity(velocity) {
			
			initParticles(m_grid_resolution);
		}

		void initParticles(int);
		// computeBoundingBox(mesh); // <--- Add this for generating bounds on the mesh when mesh loading is added
		
		std::vector<DEMParticle> getParticles() { return particles; };
		std::vector<DEMParticle> getParticlesInWorldSpace();
		Scalar* convertParticlePositionsToNakedArray();

	private:
		unsigned int m_id;

		int m_n_particles;
		Vec3f m_position; // Position of entity in World Space (at its upper left bounding box corner)
		Vec3f m_velocity;

		Scalar m_size;

		int m_grid_resolution; // The mesh is divided into particles based on this resolution

		std::vector<DEMParticle> particles;

	};

}

#endif // _STARDUST_ENTITY_HEADER_