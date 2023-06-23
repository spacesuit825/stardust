// Internal
#include "engine.hpp"
#include "cuda/cuda_utils.hpp"

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

// External
#include <json.hpp>

using json = nlohmann::json;

#define SQR(x) ((x) * (x)) // Squared function helper

namespace STARDUST {

	void DEMEngine::loadJSONSetup(std::string filepath) {
		
		std::ifstream jsonfile(filepath);
		std::stringstream buffer;
		buffer << jsonfile.rdbuf();
		auto data = json::parse(buffer);

		// Set domain size in metres
		// TODO: Add option for unbounded
		m_domain = (Scalar)data["scene"]["domain"];

		// Load a series of entities from JSON
		json entities = data["scene"]["entities"];
		for (auto entity_data : entities) {
			if (entity_data["type"] == "generic") {
				float4 pos = make_float4(
					(Scalar)entity_data["position"][0],
					(Scalar)entity_data["position"][1],
					(Scalar)entity_data["position"][2],
					0.0f
				);

				float4 vel = make_float4(
					(Scalar)entity_data["velocity"][0],
					(Scalar)entity_data["velocity"][1],
					(Scalar)entity_data["velocity"][2],
					0.0f
				);

				Scalar size = (Scalar)entity_data["size"];

				int length = m_entities.size();

				DEMEntity entity = DEMEntity(length + 1, 5, size, size, pos, vel);

				m_num_particles += entity.getParticles().size();
				m_num_entities += 1;

				m_entities.push_back(entity);
			} else if (entity_data["type"] == "mesh") {

			}
		}
	}

	void DEMEngine::prepArrays() {

		// Create host arrays

		// For Particles
		h_particle_position_ptr = new float4[m_num_particles * sizeof(float4)];
		h_particle_velocity_ptr = new float4[m_num_particles * sizeof(float4)];
		h_particle_forces_ptr = new float4[m_num_particles * sizeof(float4)];
		h_particle_mass_ptr = new float[m_num_particles * sizeof(float)];
		h_particle_to_rigid_idx_ptr = new int[m_num_particles * sizeof(int)];

		// For Rigid Bodies
		h_rigid_body_position_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_velocity_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_mass_ptr = new float[m_num_entities * sizeof(float)];
		h_rigid_body_forces_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_torques_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_linear_momentum_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_angular_momentum_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_quaternion_ptr = new float4[m_num_entities * sizeof(float4)];
		h_rigid_body_inertia_tensor_ptr = new float9[m_num_entities * sizeof(float9)];

		int offset = 0;
		for (int i = 0; i < m_entities.size(); i++) {
			// Iterate through entities and convert to naked arrays
			DEMEntity entity = m_entities[i];
			std::vector<DEMParticle> particles = entity.getParticles();

			h_rigid_body_position_ptr[i] = entity.position;
			h_rigid_body_velocity_ptr[i] = entity.velocity;
			h_rigid_body_mass_ptr[i] = entity.mass;

			// Initialise the starting dynamic values (remember: quaternion of 0 rotation is (1, 0, 0, 0))
			h_rigid_body_forces_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			h_rigid_body_torques_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			h_rigid_body_linear_momentum_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			h_rigid_body_angular_momentum_ptr[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			
			h_rigid_body_quaternion_ptr[i] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);

			int size = particles.size();

			// Create the terms for the inertia tensor
			float Ixx, Iyy, Izz, Ixy, Ixz, Iyz;
			Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0f;

			for (int j = 0; j < size; j++) {
				// Iterate through particles and convert to arrays
				DEMParticle particle = particles[j];

				h_particle_position_ptr[j + offset] = particle.position;
				h_particle_velocity_ptr[j + offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				h_particle_forces_ptr[j + offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				h_particle_mass_ptr[j + offset] = particle.mass;
				h_particle_to_rigid_idx_ptr[j + offset] = i;

				float4 relative_position = particle.position - entity.COM;

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
	}

	void DEMEngine::transferDataToDevice() {
		if (is_first_step) {
			// Mallocate the device arrays

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
				(void**)&d_particle_to_rigid_idx_ptr,
				m_num_particles * sizeof(int)
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
				d_particle_to_rigid_idx_ptr,
				h_particle_to_rigid_idx_ptr,
				m_num_particles * sizeof(int),
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

			std::cout << "Data Transfer to Device Successful!\n";
		}
	}
}