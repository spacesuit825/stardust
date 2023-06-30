#ifndef _STARDUST_ENGINE_HEADER_
#define _STARDUST_ENGINE_HEADER_

// Internal
#include "types.hpp"
#include "entity.hpp"

// CUDA
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvfunctional>

namespace STARDUST {

	class DEMEngine {
		// Class to setup and initate simulations (only handles host events)
		// Runs in parallel to DEMKernel.cu at runtime, passes events to the kernel
		// for user interaction
	public:

		DEMEngine(Scalar test) : m_domain(test) {
			
			// Computational grid pointers
			h_grid_ptr = nullptr;
			h_sphere_ptr = nullptr;
			h_grid_temp_ptr = nullptr;
			h_sphere_temp_ptr = nullptr;
			h_radices_ptr = nullptr;
			h_radix_sums_ptr = nullptr;

			d_grid_ptr = nullptr;
			d_sphere_ptr = nullptr;
			d_grid_temp_ptr = nullptr;
			d_sphere_temp_ptr = nullptr;
			d_radices_ptr = nullptr;
			d_radix_sums_ptr = nullptr;

			// Mass of rigid body
			h_rigid_body_mass_ptr = nullptr;
			d_rigid_body_mass_ptr = nullptr;

			// Position of the rigid body
			h_rigid_body_position_ptr = nullptr;
			d_rigid_body_position_ptr = nullptr;

			// Velocity of rigid body
			h_rigid_body_velocity_ptr = nullptr;
			d_rigid_body_velocity_ptr = nullptr;

			// Linear momentum of rigid body
			h_rigid_body_linear_momentum_ptr = nullptr;
			d_rigid_body_linear_momentum_ptr = nullptr;

			// Angular momentum of rigid body
			h_rigid_body_angular_momentum_ptr = nullptr;
			d_rigid_body_angular_momentum_ptr = nullptr;

			// Quaternion of rigid body
			h_rigid_body_quaternion_ptr = nullptr;
			d_rigid_body_quaternion_ptr = nullptr;

			// Inertial matrix of the rigid body
			h_rigid_body_inertia_tensor_ptr = nullptr;
			d_rigid_body_inertia_tensor_ptr = nullptr;

			// Forces on the rigid body
			h_rigid_body_forces_ptr = nullptr;
			d_rigid_body_forces_ptr = nullptr;

			// Torques on the rigid body
			h_rigid_body_torques_ptr = nullptr;
			d_rigid_body_torques_ptr = nullptr;


			// Host and Device Arrays for Particles

			// Particle to rigid body index
			h_particle_to_rigid_idx_ptr = nullptr;
			d_particle_to_rigid_idx_ptr = nullptr;

			// Position of particles
			h_particle_position_ptr = nullptr;
			d_particle_position_ptr = nullptr;

			// Velocity of particles
			h_particle_velocity_ptr = nullptr;
			d_particle_velocity_ptr = nullptr;

			// Forces on particles
			h_particle_forces_ptr = nullptr;
			d_particle_forces_ptr = nullptr;

			// Mass of the particle
			h_particle_mass_ptr = nullptr;
			d_particle_mass_ptr = nullptr;

			// Size of the particles
			h_particle_size_ptr = nullptr;
			d_particle_size_ptr = nullptr;

			is_first_step = true;

			m_num_entities = 0;
			m_num_particles = 0;
		};

		//void loadXMLSetup(std::string); // Loads setup XML file and creates the scene
		void loadJSONSetup(std::string); // Loads setup JSON file and creates the scene

		//void createEntity(int, std::string, int); // Load mesh from file and create n entities // n, filepath, grid_resolution
		//void createEntity(int, Scalar, int); // Create n generic entities from basic inputs // n, size of cube, grid_resolution

		//void cloneEntity(int); // Clone a specific entity n times, spacing is randomised by the BB of the entity
		void setUpGrid();

		void prepArrays(); // Prepare std::vectors for CUDA by translating into naked arrays
		void transferDataToDevice();

		void step(Scalar);

		~DEMEngine() {

			std::cout << "Destroying Engine...\n";

			// Computational grid pointers
			free(h_grid_ptr);
			free(h_sphere_ptr);
			free(h_grid_temp_ptr);
			free(h_sphere_temp_ptr);
			free(h_radices_ptr);
			free(h_radix_sums_ptr);

			cudaFree(d_grid_ptr);
			cudaFree(d_sphere_ptr);
			cudaFree(d_grid_temp_ptr);
			cudaFree(d_sphere_temp_ptr);
			cudaFree(d_radices_ptr);
			cudaFree(d_radix_sums_ptr);

			// Mass of the rigid body
			free(h_rigid_body_mass_ptr);
			cudaFree(d_rigid_body_mass_ptr);

			// Position of the rigid body
			free(h_rigid_body_position_ptr);
			cudaFree(d_rigid_body_position_ptr);

			// Velocity of rigid body
			free(h_rigid_body_velocity_ptr);
			cudaFree(d_rigid_body_velocity_ptr);

			// Linear momentum of rigid body
			free(h_rigid_body_linear_momentum_ptr);
			cudaFree(d_rigid_body_linear_momentum_ptr);

			// Angular momentum of rigid body
			free(h_rigid_body_angular_momentum_ptr);
			cudaFree(d_rigid_body_angular_momentum_ptr);

			// Quaternion of rigid body
			free(h_rigid_body_quaternion_ptr);
			cudaFree(d_rigid_body_quaternion_ptr);

			// Inertial matrix of the rigid body
			free(h_rigid_body_inertia_tensor_ptr);
			cudaFree(d_rigid_body_inertia_tensor_ptr);

			// Forces on the rigid body
			free(h_rigid_body_forces_ptr);
			cudaFree(d_rigid_body_forces_ptr);

			// Torques on the rigid body
			free(h_rigid_body_torques_ptr);
			cudaFree(d_rigid_body_torques_ptr);


			// Host and Device Arrays for Particles

			// Particle to rigid body index
			free(h_particle_to_rigid_idx_ptr);
			cudaFree(d_particle_to_rigid_idx_ptr);

			// Position of particles
			free(h_particle_position_ptr);
			cudaFree(d_particle_position_ptr);

			// Velocity of particles
			free(h_particle_velocity_ptr);
			cudaFree(d_particle_velocity_ptr);

			// Forces on particles
			free(h_particle_forces_ptr);
			cudaFree(d_particle_forces_ptr);

			// Mass of particles
			free(h_particle_mass_ptr);
			cudaFree(d_particle_mass_ptr);

			//Size of particles
			free(h_particle_size_ptr);
			cudaFree(d_particle_size_ptr);
		};

		int getEntityLength() { return m_entities.size(); };
		int getNumberOfSpheres() { return m_num_particles; };

		bool is_first_step;

	private:

		Scalar m_domain;

		// Make all host and device vectors 4 long if possible to coaleacse reads

		unsigned int* h_temp_ptr;
		unsigned int* d_temp_ptr;

		// Host and Device Arrays for the Computational Grid
		int* h_grid_ptr;
		int* h_sphere_ptr;
		int* h_grid_temp_ptr;
		int* h_sphere_temp_ptr;
		int* h_radices_ptr;
		int* h_radix_sums_ptr;

		int* d_grid_ptr;
		int* d_sphere_ptr;
		int* d_grid_temp_ptr;
		int* d_sphere_temp_ptr;
		int* d_radices_ptr;
		int* d_radix_sums_ptr;

		// Host and Device Arrays for Rigid Bodies (entities)
		// Mass of the rigid body
		float* h_rigid_body_mass_ptr;
		float* d_rigid_body_mass_ptr;

		// Position of the rigid body
		float4* h_rigid_body_position_ptr;
		float4* d_rigid_body_position_ptr;

		// Velocity of rigid body
		float4* h_rigid_body_velocity_ptr;
		float4* d_rigid_body_velocity_ptr;

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


		int m_num_entities;
		int m_num_particles;

		unsigned int m_cell_size;

		std::vector<DEMParticle> m_entities;
	};
}



#endif // _STARDUST_ENGINE_HEADER_