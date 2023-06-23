#include "entity.hpp"
#include "types.hpp"
#include "helper_math.hpp"

namespace STARDUST {
	
	void DEMEntity::initParticles(int grid_resolution) {
		// For now we only deal with a cube
		Scalar length = m_size; // m

		// Cast rays in the future
		Scalar dx = length / grid_resolution; // dx is the radius of particle

		// 0, 0, 0 is upper left corner of the cube

		int n_particles = grid_resolution * grid_resolution * grid_resolution;
		
		for (int i = 0; i <= grid_resolution; i++) {
			for (int j = 0; j <= grid_resolution; j++) {
				for (int k = 0; k <= grid_resolution; k++) {

					DEMParticle particle;

					particle.position = make_float4(i * dx, j * dx, k * dx, 0.0f);
					particle.size = dx;
					particle.mass = mass / n_particles; // Assume equal mass distribution

					particles.push_back(particle);

				}
			}
		}
	}

	void DEMEntity::getCenterOfMass() {
		float4 COM = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		for (int i = 0; i < particles.size(); i++) {
			COM += particles[i].position;
		}

		COM = COM / (float)particles.size();
	}

	void DEMEntity::setParticlesInWorldSpace() {

		for (int i = 0; i < particles.size(); i++) {

			DEMParticle particle = particles[i];

			float4 pos = particle.position;

			// Shift reference location to the COM
			float4 COM_pos = pos - COM;

			// Transmute position into world space, no rotation here so simple addition will work
			float4 world_pos = position - COM_pos;
			
			particle.position = world_pos;
		}
	}

	Scalar* DEMEntity::convertParticlePositionsToNakedArray() {

		Scalar* h_position_ptr = new Scalar[3 * particles.size() * sizeof(Scalar)];

		for (int i = 0; i < particles.size(); i++) {

			/*h_position_ptr[3 * i + 0] = particles[i].position(0);
			h_position_ptr[3 * i + 1] = particles[i].position(1);
			h_position_ptr[3 * i + 2] = particles[i].position(2);*/
		}

		return h_position_ptr;
	}





}