#ifndef _STARDUST_ENTITY_HANDLER_HEADER_
#define _STARDUST_ENTITY_HANDLER_HEADER_

#include "entity.hpp"

#include <vector>

namespace STARDUST {

	class EntityHandler {
		
	public:

		EntityHandler() {

			initiatePointers();

			n_primitives = 0;

			n_particles = 0;
			n_spheres = 0;

			n_meshes = 0;
		};

		int getNumberOfSpheres() { return n_spheres; };
		int getNumberOfParticles() { return particles.size(); };
		int getNumberOfMeshes() { return meshes.size(); };
		int getNumberOfPrimitives() { return n_primitives; };

		void addEntity(DEMParticle);
		void addEntity(DEMMesh);
		void removeEntity(int, int);

		void initiatePointers();
		void prepareArrays();
		void allocateCUDA();
		void transferDataToDevice();

		void addEntityRuntime(DEMParticle);
		void addEntityRuntime(DEMMesh);
		void removeEntityRuntime(int, int);

		// Pointers for particles and spheres
		int* h_entity_start_ptr;
		int* d_entity_start_ptr;

		int* h_entity_length_ptr;
		int* d_entity_length_ptr;

		int* h_particle_sorted_idx_ptr;
		int* d_particle_sorted_idx_ptr;

		int* h_particle_idx_ptr;
		int* d_particle_idx_ptr;

		// Mass of the rigid body
		float* h_rigid_body_mass_ptr;
		float* d_rigid_body_mass_ptr;

		// Position of the rigid body
		float4* h_rigid_body_position_ptr;
		float4* d_rigid_body_position_ptr;

		// Velocity of rigid body
		float4* h_rigid_body_velocity_ptr;
		float4* d_rigid_body_velocity_ptr;

		float4* h_rigid_body_angular_velocity_ptr;
		float4* d_rigid_body_angular_velocity_ptr;

		// Linear momentum of rigid body
		float4* h_rigid_body_linear_momentum_ptr;
		float4* d_rigid_body_linear_momentum_ptr;

		// Angular momentum of rigid body
		float4* h_rigid_body_angular_momentum_ptr;
		float4* d_rigid_body_angular_momentum_ptr;

		// Quaternion of rigid body
		float4* h_rigid_body_quaternion_ptr;
		float4* d_rigid_body_quaternion_ptr;

		// Inertial matrix of the rigid body
		float9* h_rigid_body_inertia_tensor_ptr;
		float9* d_rigid_body_inertia_tensor_ptr;

		// Forces on the rigid body
		float4* h_rigid_body_forces_ptr;
		float4* d_rigid_body_forces_ptr;

		// Torques on the rigid body
		float4* h_rigid_body_torques_ptr;
		float4* d_rigid_body_torques_ptr;


		// Host and Device Arrays for Particles
		// Particle to rigid body index
		int* h_particle_to_rigid_idx_ptr;
		int* d_particle_to_rigid_idx_ptr;

		// Position of particles
		float4* h_particle_position_ptr;
		float4* d_particle_position_ptr;

		// Velocity of particles
		float4* h_particle_velocity_ptr;
		float4* d_particle_velocity_ptr;

		// Forces on particles
		float4* h_particle_forces_ptr;
		float4* d_particle_forces_ptr;

		// Mass of particles
		float* h_particle_mass_ptr;
		float* d_particle_mass_ptr;

		float* h_particle_size_ptr;
		float* d_particle_size_ptr;

		float4* h_particle_relative_position_ptr;
		float4* d_particle_relative_position_ptr;

		float4* h_particle_init_relative_position_ptr;
		float4* d_particle_init_relative_position_ptr;

	private:

		int n_primitives;

		int n_particles;
		int n_spheres;

		int n_meshes;
		
		std::vector<DEMParticle> particles;
		std::vector<DEMMesh> meshes;
	};

}





#endif // _STARDUST_ENTITY_HANDLER_HEADER_