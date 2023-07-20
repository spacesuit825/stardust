// Internal
#include "engine.hpp"
#include "cuda/cuda_utils.hpp"
#include "cuda/collision_detection.cuh"

// C++
#include <string>
#include <iostream>
#include <fstream>

// CUDA
#include <vector_types.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvfunctional>
#include <../../glew/include/GL/glew.h>
#include <cuda_gl_interop.h>

// External
//#include <json.hpp>

//using json = nlohmann::json;

#define SQR(x) ((x) * (x)) // Squared function helper



namespace STARDUST {

#pragma warning (disable : 4068 ) /* disable unknown pragma warnings */

	// Handle CUDA/OpenGL interop

	// Bind vbo to GPU buffers to prevent repeat data transmissions
	void DEMEngine::bindGLBuffers(const GLuint pos_buffer) {
		// To avoid the high cost of registering a cuda/GL buffer we initialise with the max particle count
		CUDA_ERR_CHECK(cudaGraphicsGLRegisterBuffer(&vbo_position, pos_buffer, cudaGraphicsMapFlagsWriteDiscard));
	}

	void DEMEngine::unbindGLBuffers(const GLuint pos_buffer) {
		CUDA_ERR_CHECK(cudaGraphicsUnregisterResource(vbo_position));
	}

	void DEMEngine::writeGLPosBuffer() {
		float4* bufptr;
		size_t size;

		CUDA_ERR_CHECK(cudaGraphicsMapResources(1, &vbo_position, NULL));
		CUDA_ERR_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&bufptr, &size, vbo_position));
		CUDA_ERR_CHECK(cudaMemcpy(bufptr,
			entity_handler.d_particle_position_ptr,
			sizeof(float4) * entity_handler.getNumberOfSpheres(),
			cudaMemcpyDeviceToDevice
		));
		CUDA_ERR_CHECK(cudaGraphicsUnmapResources(1, &vbo_position, NULL));
	}

	// Map to the cuda buffer so OpenGL can access the data via device pointer
	void DEMEngine::writeGLBuffers() {
		writeGLPosBuffer();
	}

	// WHY IS THIS NOT WORKING??? //
	//void DEMEngine::loadJSONSetup(std::string filepath) {
	//	
	//	std::ifstream jsonfile(filepath);
	//	std::stringstream buffer;
	//	buffer << jsonfile.rdbuf();
	//	auto data = json::parse(buffer);

	//	// Set domain size in metres
	//	// TODO: Add option for unbounded
	//	m_domain = (Scalar)data["scene"]["domain"];

	//	// Load a series of entities from JSON
	//	json entities = data["scene"]["entities"];
	//	for (auto entity_data : entities) {
	//		if (entity_data["type"] == "generic") {
	//			float4 pos = make_float4(
	//				(Scalar)entity_data["position"][0],
	//				(Scalar)entity_data["position"][1],
	//				(Scalar)entity_data["position"][2],
	//				0.0f
	//			);

	//			float4 vel = make_float4(
	//				(Scalar)entity_data["velocity"][0],
	//				(Scalar)entity_data["velocity"][1],
	//				(Scalar)entity_data["velocity"][2],
	//				0.0f
	//			);

	//			Scalar size = (Scalar)entity_data["size"];

	//			int length = m_entities.size();

	//			DEMParticle entity = DEMParticle(length + 1, 10, size, size, pos, vel);

	//			m_num_particles += entity.getParticles().size();
	//			m_num_entities += 1;

	//			m_entities.push_back(entity);
	//		} else if (entity_data["type"] == "mesh") {

	//		}
	//	}
	//}

	void DEMEngine::addParticle(DEMParticle particle) {
		m_entities.push_back(particle);

		m_num_particles += particle.getParticles().size();
		m_num_entities += 1;
	}

	void DEMEngine::addMesh(DEMMesh mesh) {
		m_meshes.push_back(mesh);

		m_num_meshes += 1;
	}

	void DEMEngine::setUpGrid() {
		int num_particles = m_num_particles;

		m_cell_size = num_particles * 9 * sizeof(int);

	}

	void DEMEngine::prepArrays() {

		// Create host arra

		// For Particles
		h_particle_position_ptr = new float4[m_num_particles * sizeof(float4)];
		h_particle_velocity_ptr = new float4[m_num_particles * sizeof(float4)];
		h_particle_forces_ptr = new float4[m_num_particles * sizeof(float4)];
		h_particle_mass_ptr = new float[m_num_particles * sizeof(float)];
		h_particle_size_ptr = new float[m_num_particles * sizeof(float)];
		h_particle_to_rigid_idx_ptr = new int[m_num_particles * sizeof(int)];
		h_particle_init_relative_position_ptr = new float4[m_num_particles * sizeof(float4)];
		h_particle_relative_position_ptr = new float4[m_num_particles * sizeof(float4)];

		// For Rigid Bodies
		h_rigid_body_position_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_velocity_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_angular_velocity_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_mass_ptr = new float[m_num_entities * sizeof(float)];
		h_rigid_body_forces_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_torques_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_linear_momentum_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_angular_momentum_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_quaternion_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_inertia_tensor_ptr = new float9[m_num_entities * sizeof(float9)];

		h_entity_start_ptr = new int[m_num_entities * sizeof(int)];
		h_entity_length_ptr = new int[m_num_entities * sizeof(int)];

		int offset = 0;
		for (int i = 0; i < m_entities.size(); i++) {
			// Iterate through entities and convert to naked arrays
			DEMParticle entity = m_entities[i];
			std::vector<DEMSphere> particles = entity.getParticles();

			h_rigid_body_position_ptr[i] = entity.position;
			h_rigid_body_velocity_ptr[i] = entity.velocity;
			h_rigid_body_angular_velocity_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			h_rigid_body_mass_ptr[i] = entity.mass;

			// Initialise the starting dynamic values (remember: quaternion of 0 rotation is (1, 0, 0, 0))
			h_rigid_body_forces_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			h_rigid_body_torques_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			h_rigid_body_linear_momentum_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			h_rigid_body_angular_momentum_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			
			h_rigid_body_quaternion_ptr[i] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);

			int size = particles.size();

			h_entity_start_ptr[i] = offset;
			h_entity_length_ptr[i] = size;

			// Create the terms for the inertia tensor
			float Ixx, Iyy, Izz, Ixy, Ixz, Iyz;
			Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0f;

			for (int j = 0; j < size; j++) {
				// Iterate through particles and convert to arrays
				DEMSphere particle = particles[j];

				h_particle_position_ptr[j + offset] = particle.position;
				h_particle_velocity_ptr[j + offset] = entity.velocity;
				h_particle_forces_ptr[j + offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				h_particle_mass_ptr[j + offset] = particle.mass;
				h_particle_size_ptr[j + offset] = particle.size;
				h_particle_to_rigid_idx_ptr[j + offset] = i;
				h_particle_init_relative_position_ptr[j + offset] = particle.position - entity.position;
				h_particle_relative_position_ptr[j + offset] = particle.position - entity.position;

				float4 relative_position = particle.position - entity.position;

				Ixx += particle.mass * (SQR(relative_position.y) + SQR(relative_position.z));
				Iyy += particle.mass * (SQR(relative_position.x) + SQR(relative_position.z));
				Izz += particle.mass * (SQR(relative_position.x) + SQR(relative_position.y));
				Ixy += -particle.mass * relative_position.x * relative_position.y;
				Ixz += -particle.mass * relative_position.x * relative_position.z;
				Iyz += -particle.mass * relative_position.y * relative_position.z;
			}

			// Create the inertia tensor (3x3 symmetric)
			float9 inertia_tensor = { Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz };
			h_rigid_body_inertia_tensor_ptr[i] = inertia_tensor;

			offset += size;
		}

		//h_mesh_vertex_ptr = new float4[m_num_triangles * 3 * sizeof(float4)];
		//h_mesh_index_ptr = new int[m_num_triangles * 3 * sizeof(int)];
		//h_mesh_quat_ptr = new float4[m_meshes.size() * sizeof(float4)];

		// Prep mesh arrays for GPU transfer
		/*offset = 0;
		for (int i = 0; i < m_meshes.size(); i++) {

			DEMMesh mesh = m_meshes[i];

			h_mesh_start_ptr[i] = offset;
			h_mesh_length_ptr[i] = mesh.n_triangles * 3;

			h_mesh_quat_ptr[i] = mesh.quat;

			for (int j = 0; i < mesh.n_triangles * 3; j++) {
				
				h_mesh_vertex_ptr[j + offset] = mesh.vertices[j];
				h_mesh_index_ptr[j + offset] = mesh.indicies[j] + offset;

			}

			offset += (mesh.n_triangles * 3);
		}*/
	}

	void DEMEngine::transferDataToDevice() {
		if (is_first_step) {
			// Mallocate the device arrays

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_temp_ptr,
				2 * sizeof(unsigned int)
			));

			// Allocate computational grid arrays
			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_grid_ptr,
				m_num_particles * 9 * sizeof(uint32_t)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_grid_temp_ptr,
				m_num_particles * 9 * sizeof(uint32_t)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_sphere_ptr,
				m_num_particles * 9 * sizeof(uint32_t)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_sphere_temp_ptr,
				m_num_particles * 9 * sizeof(uint32_t)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_radices_ptr,
				NUM_BLOCKS * NUM_RADICES
				* GROUPS_PER_BLOCK * sizeof(uint32_t)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_radix_sums_ptr,
				NUM_RADICES * sizeof(uint32_t)
			));

			// Allocate particle arrays
			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_particle_position_ptr,
				m_num_particles * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_particle_velocity_ptr,
				m_num_particles * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_particle_forces_ptr,
				m_num_particles * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_particle_mass_ptr,
				m_num_particles * sizeof(float)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_particle_size_ptr,
				m_num_particles * sizeof(float)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_particle_to_rigid_idx_ptr,
				m_num_particles * sizeof(int)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_particle_relative_position_ptr,
				m_num_particles * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_particle_init_relative_position_ptr,
				m_num_particles * sizeof(float4)
			));

			// Allocate rigid body arrays
			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_position_ptr,
				m_num_entities * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_velocity_ptr,
				m_num_entities * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_angular_velocity_ptr,
				m_num_entities * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_mass_ptr,
				m_num_entities * sizeof(float)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_forces_ptr,
				m_num_entities * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_torques_ptr,
				m_num_entities * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_linear_momentum_ptr,
				m_num_entities * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_angular_momentum_ptr,
				m_num_entities * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_quaternion_ptr,
				m_num_entities * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_rigid_body_inertia_tensor_ptr,
				m_num_entities * sizeof(float9)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_entity_start_ptr,
				m_num_entities * sizeof(int)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_entity_length_ptr,
				m_num_entities * sizeof(int)
			));

			/*CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_mesh_vertex_ptr,
				m_num_triangles * 3 * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_mesh_index_ptr,
				m_num_triangles * 3 * sizeof(int)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_mesh_quat_ptr,
				m_num_meshes * sizeof(float4)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_mesh_start_ptr,
				m_num_meshes * sizeof(int)
			));

			CUDA_ERR_CHECK(cudaMalloc(
				(void**)&d_mesh_length_ptr,
				m_num_meshes * sizeof(int)
			));*/

			std::cout << "Allocation on Device Successful!\n";

		}
		else {

			CUDA_ERR_CHECK(cudaDeviceSynchronize());

			// Transfer data to device

			// Transfer particle arrays from host to device
			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_particle_position_ptr,
				h_particle_position_ptr,
				m_num_particles * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_particle_velocity_ptr,
				h_particle_velocity_ptr,
				m_num_particles * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_particle_forces_ptr,
				h_particle_forces_ptr,
				m_num_particles * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_particle_mass_ptr,
				h_particle_mass_ptr,
				m_num_particles * sizeof(float),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_particle_size_ptr,
				h_particle_size_ptr,
				m_num_particles * sizeof(float),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_particle_to_rigid_idx_ptr,
				h_particle_to_rigid_idx_ptr,
				m_num_particles * sizeof(int),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_particle_relative_position_ptr,
				h_particle_relative_position_ptr,
				m_num_particles * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_particle_init_relative_position_ptr,
				h_particle_init_relative_position_ptr,
				m_num_particles * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			// Transfer rigid body arrays from host to device
			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_position_ptr,
				h_rigid_body_position_ptr,
				m_num_entities * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_velocity_ptr,
				h_rigid_body_velocity_ptr,
				m_num_entities * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_angular_velocity_ptr,
				h_rigid_body_angular_velocity_ptr,
				m_num_entities * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_mass_ptr,
				h_rigid_body_mass_ptr,
				m_num_entities * sizeof(float),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_forces_ptr,
				h_rigid_body_forces_ptr,
				m_num_entities * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_torques_ptr,
				h_rigid_body_torques_ptr,
				m_num_entities * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_linear_momentum_ptr,
				h_rigid_body_linear_momentum_ptr,
				m_num_entities * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_angular_momentum_ptr,
				h_rigid_body_angular_momentum_ptr,
				m_num_entities * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_quaternion_ptr,
				h_rigid_body_quaternion_ptr,
				m_num_entities * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_rigid_body_inertia_tensor_ptr,
				h_rigid_body_inertia_tensor_ptr,
				m_num_entities * sizeof(float9),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_entity_start_ptr,
				h_entity_start_ptr,
				m_num_entities * sizeof(int),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_entity_length_ptr,
				h_entity_length_ptr,
				m_num_entities * sizeof(int),
				cudaMemcpyHostToDevice)
			);

			/*CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_mesh_vertex_ptr,
				h_mesh_vertex_ptr,
				m_num_triangles * 3 * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_mesh_index_ptr,
				h_mesh_index_ptr,
				m_num_triangles * 3 * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_mesh_quat_ptr,
				h_mesh_quat_ptr,
				m_num_meshes * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_mesh_start_ptr,
				h_mesh_start_ptr,
				m_num_meshes * sizeof(float4),
				cudaMemcpyHostToDevice)
			);

			CUDA_ERR_CHECK(cudaMemcpyAsync(
				d_mesh_length_ptr,
				h_mesh_length_ptr,
				m_num_meshes * sizeof(float4),
				cudaMemcpyHostToDevice)
			);*/

			std::cout << "Data Transfer to Device Successful!\n";
		}
	}

	void DEMEngine::add(DEMParticle particle) {
		entity_handler.addEntity(particle);
	}

	void DEMEngine::prep() {
		entity_handler.prepareArrays();
	}

	void DEMEngine::transfer() {
		entity_handler.allocateCUDA();
		entity_handler.transferDataToDevice();

		collision_handler.allocateCUDA(entity_handler.getNumberOfSpheres(), 10);
	}

	void DEMEngine::update(float dt) {

		collision_handler.processCollisions(entity_handler.d_particle_position_ptr, entity_handler.d_particle_size_ptr, entity_handler.getNumberOfSpheres());

	}
}