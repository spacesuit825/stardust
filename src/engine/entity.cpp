#include "entity.hpp"
#include "types.hpp"
#include "helper_math.hpp"

namespace STARDUST {
	
	void DEMParticle::initParticles(int grid_resolution) {
		// For now we only deal with a cube
		Scalar length = m_size; // m

		// Cast rays in the future
		int num_particles = floor(length / diameter);
		std::cout << "Number of Particles: " << num_particles << "\n";
		float dx = length / num_particles;

		// 0, 0, 0 is upper left corner of the cube
		
		for (int i = 1; i <= grid_resolution; i++) {
			for (int j = 1; j <= grid_resolution; j++) {
				for (int k = 1; k <= grid_resolution; k++) {

					DEMSphere particle;

					particle.position = make_float4(i * dx, j * dx, k * dx, 0.0f);
					particle.size = diameter;
					particle.mass = mass / num_particles; // Assume equal mass distribution

					particles.push_back(particle);

				}
			}
		}
	}

	void DEMParticle::getCenterOfMass() {
		float4 COM = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		for (int i = 0; i < particles.size(); i++) {
			COM += particles[i].position;
		}

		COM = COM / (float)particles.size();
		printf("COM: %.3f, %.3f, %.3f \n", COM.x, COM.y, COM.z);
	}

	void DEMParticle::setParticlesInWorldSpace() {

		for (int i = 0; i < particles.size(); i++) {

			DEMSphere& particle = particles[i];

			float4 pos = particle.position;

			// Shift reference location to the COM
			float4 COM_pos = pos - COM;

			// Transmute position into world space, no rotation here so simple addition will work
			float4 world_pos = pos - position;
			
			particle.position = world_pos;
		}
	}

	Scalar* DEMParticle::convertParticlePositionsToNakedArray() {

		Scalar* h_position_ptr = new Scalar[3 * particles.size() * sizeof(Scalar)];

		for (int i = 0; i < particles.size(); i++) {

			/*h_position_ptr[3 * i + 0] = particles[i].position(0);
			h_position_ptr[3 * i + 1] = particles[i].position(1);
			h_position_ptr[3 * i + 2] = particles[i].position(2);*/
		}

		return h_position_ptr;
	}





}