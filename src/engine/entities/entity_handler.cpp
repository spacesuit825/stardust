#include "entity_handler.hpp"
#include "./cuda/cuda_utils.hpp"

#define SQR(x) ((x) * (x)) // Squared function helper

namespace STARDUST {

	void EntityHandler::prepareArrays() {
		// For Particles
		h_particle_position_ptr = new float4[n_spheres * sizeof(float4)];
		h_particle_velocity_ptr = new float4[n_spheres * sizeof(float4)];
		h_particle_forces_ptr = new float4[n_spheres * sizeof(float4)];
		h_particle_mass_ptr = new float[n_spheres * sizeof(float)];
		h_particle_size_ptr = new float[n_spheres * sizeof(float)];
		h_particle_to_rigid_idx_ptr = new int[n_spheres * sizeof(int)];
		h_particle_init_relative_position_ptr = new float4[n_spheres * sizeof(float4)];
		h_particle_relative_position_ptr = new float4[n_spheres * sizeof(float4)];

		// For Rigid Bodies
		h_rigid_body_position_ptr = new float4[n_particles * sizeof(float4)];
		h_rigid_body_velocity_ptr = new float4[n_particles * sizeof(float4)];
		h_rigid_body_angular_velocity_ptr = new float4[n_particles * sizeof(float4)];
		h_rigid_body_mass_ptr = new float[n_particles * sizeof(float)];
		h_rigid_body_forces_ptr = new float4[n_particles * sizeof(float4)];
		h_rigid_body_torques_ptr = new float4[n_particles * sizeof(float4)];
		h_rigid_body_linear_momentum_ptr = new float4[n_particles * sizeof(float4)];
		h_rigid_body_angular_momentum_ptr = new float4[n_particles * sizeof(float4)];
		h_rigid_body_quaternion_ptr = new float4[n_particles * sizeof(float4)];
		h_rigid_body_inertia_tensor_ptr = new float9[n_particles * sizeof(float9)];

		h_particle_idx_ptr = new int[n_spheres * sizeof(int)];
		h_particle_sorted_idx_ptr = new int[n_spheres * sizeof(int)];

		h_entity_start_ptr = new int[n_particles * sizeof(int)];
		h_entity_length_ptr = new int[n_particles * sizeof(int)];

		int offset = 0;
		for (int i = 0; i < particles.size(); i++) {
			// Iterate through entities and convert to naked arrays
			DEMParticle entity = particles[i];
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

				//printf("Particle: %d, position: %.3f, %.3f, %.3f", j, particle.position.x, particle.position.y, particle.position)

				h_particle_sorted_idx_ptr[j + offset] = j + offset;
				h_particle_idx_ptr[j + offset] = j + offset;
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
	}

	// Add or remove entities
	void EntityHandler::addEntity(DEMParticle particle) {
		particles.push_back(particle);

		n_primitives += particle.getParticles().size();
		n_spheres += particle.getParticles().size();
		n_particles += 1;
	}
	void EntityHandler::addEntityRuntime(DEMParticle particle) {}

	void EntityHandler::addEntity(DEMMesh mesh) {
		meshes.push_back(mesh);

		n_primitives += mesh.n_triangles;
		n_meshes += 1;
	}
	void EntityHandler::addEntityRuntime(DEMMesh mesh) {}

	void EntityHandler::removeEntity(int type, int idx) {}
	void EntityHandler::removeEntityRuntime(int type, int idx) {}

	// Initiate pointers for CUDA arrays
	void EntityHandler::initiatePointers() {
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
	}

	// Mallocate the required space on the GPU
	void EntityHandler::allocateCUDA() {

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_sorted_idx_ptr,
			n_spheres * sizeof(int)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_idx_ptr,
			n_spheres * sizeof(int)
		));
		
		// Allocate particle arrays
		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_position_ptr,
			n_spheres * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_velocity_ptr,
			n_spheres * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_forces_ptr,
			n_spheres * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_mass_ptr,
			n_spheres * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_size_ptr,
			n_spheres * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_to_rigid_idx_ptr,
			n_spheres * sizeof(int)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_relative_position_ptr,
			n_spheres * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_particle_init_relative_position_ptr,
			n_spheres * sizeof(float4)
		));

		// Allocate rigid body arrays
		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_position_ptr,
			n_particles * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_velocity_ptr,
			n_particles * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_angular_velocity_ptr,
			n_particles * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_mass_ptr,
			n_particles * sizeof(float)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_forces_ptr,
			n_particles * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_torques_ptr,
			n_particles * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_linear_momentum_ptr,
			n_particles * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_angular_momentum_ptr,
			n_particles * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_quaternion_ptr,
			n_particles * sizeof(float4)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_rigid_body_inertia_tensor_ptr,
			n_particles * sizeof(float9)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_entity_start_ptr,
			n_particles * sizeof(int)
		));

		CUDA_ERR_CHECK(cudaMalloc(
			(void**)&d_entity_length_ptr,
			n_particles * sizeof(int)
		));
	}
	
	// Transfer the data to the device
	void EntityHandler::transferDataToDevice() {

		CUDA_ERR_CHECK(cudaDeviceSynchronize());
		// Transfer data to device

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_sorted_idx_ptr,
			h_particle_sorted_idx_ptr,
			n_spheres * sizeof(int),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_idx_ptr,
			h_particle_idx_ptr,
			n_spheres * sizeof(int),
			cudaMemcpyHostToDevice)
		);

		// Transfer particle arrays from host to device
		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_position_ptr,
			h_particle_position_ptr,
			n_spheres * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_velocity_ptr,
			h_particle_velocity_ptr,
			n_spheres * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_forces_ptr,
			h_particle_forces_ptr,
			n_spheres * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_mass_ptr,
			h_particle_mass_ptr,
			n_spheres * sizeof(float),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_size_ptr,
			h_particle_size_ptr,
			n_spheres * sizeof(float),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_to_rigid_idx_ptr,
			h_particle_to_rigid_idx_ptr,
			n_spheres * sizeof(int),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_relative_position_ptr,
			h_particle_relative_position_ptr,
			n_spheres * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_particle_init_relative_position_ptr,
			h_particle_init_relative_position_ptr,
			n_spheres * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		// Transfer rigid body arrays from host to device
		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_position_ptr,
			h_rigid_body_position_ptr,
			n_particles * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_velocity_ptr,
			h_rigid_body_velocity_ptr,
			n_particles * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_angular_velocity_ptr,
			h_rigid_body_angular_velocity_ptr,
			n_particles * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_mass_ptr,
			h_rigid_body_mass_ptr,
			n_particles * sizeof(float),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_forces_ptr,
			h_rigid_body_forces_ptr,
			n_particles * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_torques_ptr,
			h_rigid_body_torques_ptr,
			n_particles * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_linear_momentum_ptr,
			h_rigid_body_linear_momentum_ptr,
			n_particles * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_angular_momentum_ptr,
			h_rigid_body_angular_momentum_ptr,
			n_particles * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_quaternion_ptr,
			h_rigid_body_quaternion_ptr,
			n_particles * sizeof(float4),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_rigid_body_inertia_tensor_ptr,
			h_rigid_body_inertia_tensor_ptr,
			n_particles * sizeof(float9),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_entity_start_ptr,
			h_entity_start_ptr,
			n_particles * sizeof(int),
			cudaMemcpyHostToDevice)
		);

		CUDA_ERR_CHECK(cudaMemcpyAsync(
			d_entity_length_ptr,
			h_entity_length_ptr,
			n_particles * sizeof(int),
			cudaMemcpyHostToDevice)
		);
	}
}