// Internal
#include "../engine.hpp"
#include "cuda_utils.hpp"
#include "collision_detection.cuh"

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


namespace STARDUST {

	// Template for launching kernels
	template<typename... Arguments>
	void KernelLaunch(std::string&& tag, int gs, int bs, void(*f)(Arguments...), Arguments... args) {
		f << <gs, bs >> > (args...);

		CUDA_ERR_CHECK(cudaPeekAtLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
	}





	void DEMEngine::step(Scalar timestep) {

		float cell_dim = 1000;

		int threads_per_block = 128;
		unsigned int particle_size = (m_num_particles - 1) / threads_per_block + 1;
		
		constructCells(
			m_num_particles, 
			cell_dim, 
			d_grid_ptr, 
			d_sphere_ptr, 
			d_particle_size_ptr, 
			d_particle_position_ptr, 
			threads_per_block,
			d_temp_ptr
		);

		unsigned int count;
		cudaMemcpy(&count, d_temp_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		std::cout << "Cells Occupied " << count << "\n";

		sortCells(
			d_grid_ptr,
			d_sphere_ptr,
			d_grid_temp_ptr,
			d_sphere_temp_ptr,
			d_radices_ptr,
			d_radix_sums_ptr,
			m_num_particles
		);

		collideCells(
			d_grid_ptr,
			d_sphere_ptr,
			d_particle_position_ptr,
			d_particle_velocity_ptr,
			d_particle_size_ptr,
			m_num_particles,
			d_temp_ptr,
			threads_per_block
		);

		int test;
		cudaMemcpy(&count, d_temp_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&test, d_temp_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		std::cout << "Collisions Detected " << count << "\n";
		std::cout << "Tests Performed " << count << "\n";
	}
}