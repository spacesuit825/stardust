#ifndef _STARDUST_ENGINE_HEADER_
#define _STARDUST_ENGINE_HEADER_

// Internal
#include "types.hpp"
#include "./entities/entity.hpp"
#include "./entities/entity_handler.hpp"
#include "./cuda/collisions/SAP/sap_collision.cuh"

// CUDA
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvfunctional>
#include <../../glew/include/GL/glew.h>
#include <cuda_gl_interop.h>

namespace STARDUST {

	enum CollisionEngine {
		SAP,
		SHP
	};

	enum Device {
		CPU,
		GPU
	};


	struct engineConfig {
		CollisionEngine collision_engine;
		Device device;
	};

	class DEMEngine {
		// Class to setup and initate simulations (only handles host events)
		// Runs in parallel to DEMKernel.cu at runtime, passes events to the kernel
		// for user interaction
	public:

		DEMEngine(engineConfig config, Scalar test) : m_domain(test), engineConfiguration(config) {

			entity_handler = EntityHandler();

			if (engineConfiguration.collision_engine == SAP) {
				collision_handler = SAPCollision();
			}
			else {
				std::cout << "Invalid Collision Engine!" << "\n";
			}
			
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

			// Mesh pointers

			h_mesh_start_ptr = nullptr;
			d_mesh_start_ptr = nullptr;

			h_mesh_length_ptr = nullptr;
			d_mesh_length_ptr = nullptr;

			h_mesh_vertex_ptr = nullptr; // Number of vertices long
			d_mesh_vertex_ptr = nullptr;

			h_mesh_index_ptr = nullptr; // Number of vertices long
			d_mesh_index_ptr = nullptr;

			h_mesh_quat_ptr = nullptr; // Number of meshes long
			d_mesh_quat_ptr = nullptr;

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

			h_rigid_body_angular_velocity_ptr = nullptr;
			d_rigid_body_angular_velocity_ptr = nullptr;

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

			h_particle_relative_position_ptr = nullptr;
			d_particle_relative_position_ptr = nullptr;

			h_particle_init_relative_position_ptr = nullptr;
			d_particle_init_relative_position_ptr = nullptr;

			is_first_step = true;
			spatialHashCollision = true;

			m_num_entities = 0;
			m_num_particles = 0;
		};

		// OPENGL VISUALISATION //
		void bindGLBuffers(const GLuint);
		void unbindGLBuffers(const GLuint);
		void writeGLBuffers();
		void writeGLPosBuffer();

		//void loadXMLSetup(std::string); // Loads setup XML file and creates the scene
		//void loadJSONSetup(std::string); // Loads setup JSON file and creates the scene
		void addParticle(DEMParticle);
		void addMesh(DEMMesh);

		//void createEntity(int, std::string, int); // Load mesh from file and create n entities // n, filepath, grid_resolution
		//void createEntity(int, Scalar, int); // Create n generic entities from basic inputs // n, size of cube, grid_resolution

		//void cloneEntity(int); // Clone a specific entity n times, spacing is randomised by the BB of the entity
		void setUpGrid();

		void prepArrays(); // Prepare std::vectors for CUDA by translating into naked arrays
		void transferDataToDevice();

		void cleanBuffers();
		void step(Scalar);

		void add(DEMParticle particle);
		void prep();
		void transfer();
		void update(float);

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

			free(h_mesh_index_ptr);
			free(h_mesh_vertex_ptr);
			free(h_mesh_length_ptr);
			free(h_mesh_quat_ptr);
			free(h_mesh_start_ptr);

			cudaFree(d_mesh_index_ptr);
			cudaFree(d_mesh_vertex_ptr);
			cudaFree(d_mesh_quat_ptr);
			cudaFree(d_mesh_length_ptr);
			cudaFree(d_mesh_start_ptr);

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

			free(h_rigid_body_angular_velocity_ptr);
			cudaFree(d_rigid_body_angular_velocity_ptr);


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

			free(h_particle_relative_position_ptr);
			cudaFree(d_particle_relative_position_ptr);

			free(h_particle_init_relative_position_ptr);
			cudaFree(d_particle_init_relative_position_ptr);

			cudaGraphicsUnregisterResource(vbo_position);
		};


		std::vector<DEMParticle> getEntities() { return m_entities; };
		int getEntityLength() { return m_entities.size(); };
		int getNumberOfSpheres() { return m_num_particles; };

		EntityHandler& getEntityHandler() { return entity_handler; };

		bool is_first_step;

	private:

		engineConfig engineConfiguration;
		SAPCollision collision_handler;
		EntityHandler entity_handler;

		struct cudaGraphicsResource* vbo_position;

		Scalar m_domain;

		// Make all host and device vectors 4 long if possible to coaleacse reads

		unsigned int* h_temp_ptr;
		unsigned int* d_temp_ptr;

		// Spatial Partitioning Data Pointers
		uint32_t* h_grid_ptr;
		uint32_t* h_sphere_ptr;
		uint32_t* h_grid_temp_ptr;
		uint32_t* h_sphere_temp_ptr;
		uint32_t* h_radices_ptr;
		uint32_t* h_radix_sums_ptr;

		uint32_t* d_grid_ptr;
		uint32_t* d_sphere_ptr;
		uint32_t* d_grid_temp_ptr;
		uint32_t* d_sphere_temp_ptr;
		uint32_t* d_radices_ptr;
		uint32_t* d_radix_sums_ptr;

		// SAP Data Pointers


		// Host and Device Arrays for Rigid Bodies (entities)
		// Entity trackers
		int* h_entity_start_ptr;
		int* d_entity_start_ptr;

		int* h_entity_length_ptr;
		int* d_entity_length_ptr;

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


		// Similar to the entity trackers, tracks the start and end of the meshes //
		int* h_mesh_start_ptr;
		int* d_mesh_start_ptr;

		int* h_mesh_length_ptr;
		int* d_mesh_length_ptr;

		float4* h_mesh_vertex_ptr; // Number of vertices long
		float4* d_mesh_vertex_ptr;

		int* h_mesh_index_ptr; // Number of vertices long
		int* d_mesh_index_ptr;

		float4* h_mesh_quat_ptr; // Number of meshes long
		float4* d_mesh_quat_ptr;

		int m_num_entities;
		int m_num_particles;
		int m_num_meshes;
		int m_num_triangles;

		bool spatialHashCollision;
		bool LBVHCollision;

		unsigned int m_cell_size;

		std::vector<DEMParticle> m_entities;
		std::vector<DEMMesh> m_meshes;
	};
}



#endif // _STARDUST_ENGINE_HEADER_