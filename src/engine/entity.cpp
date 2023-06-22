#include "entity.hpp"
#include "types.hpp"

namespace STARDUST {
	
	void DEMEntity::initParticles(int grid_resolution) {
		// For now we only deal with a cube
		Scalar length = m_size; // m

		// Cast rays in the future
		Scalar dx = length / grid_resolution; // dx is the radius of particle

		// 0, 0, 0 is upper left corner of the cube
		
		for (int i = 0; i <= grid_resolution; i++) {
			for (int j = 0; j <= grid_resolution; j++) {
				for (int k = 0; k <= grid_resolution; k++) {

					DEMParticle particle;

					particle.position = Vec3f(i * dx, j * dx, k * dx);
					particle.size = dx;
					particle.density = 1;

					particles.push_back(particle);

				}
			}
		}
	}

	std::vector<DEMParticle> DEMEntity::getParticlesInWorldSpace() {
		std::vector<DEMParticle> world_particles;

		for (int i = 0; i < particles.size(); i++) {

			DEMParticle particle = particles[i];

			Vec3f pos = particle.position;

			// Transmute position into world space, no rotation here so simple addition will work
			Vec3f world_pos = m_position - pos;
			
			DEMParticle world_particle;
			world_particle.position = world_pos;
			world_particle.density = particle.density;
			world_particle.size = particle.size;

			world_particles.push_back(world_particle);

		}

		return world_particles;
	}

	Scalar* DEMEntity::convertParticlePositionsToNakedArray() {

		Scalar* h_position_ptr = new Scalar[3 * particles.size() * sizeof(Scalar)];

		for (int i = 0; i < particles.size(); i++) {

			h_position_ptr[3 * i + 0] = particles[i].position[0];
			h_position_ptr[3 * i + 1] = particles[i].position[1];
			h_position_ptr[3 * i + 2] = particles[i].position[2];
		}

		return h_position_ptr;
	}





}