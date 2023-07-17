#include "mesh_loader.hpp"
#include "stl_reader.h"
#include <iostream>

// CUDA
#include <vector_types.h>
#include "cuda.h"
#include "cuda_runtime.h"

namespace STARDUST {

	void loadAndConvertMesh(const char* filename, std::vector<float4>& out_vert, std::vector<int>& out_idx, std::vector<float4>& out_norm) {
		// Intermediate storage for data
		std::vector<float> coords, normals;
		std::vector<unsigned int> indicies, solids;

		stl_reader::ReadStlFile(filename, coords, normals, indicies, solids);

		std::cout << "Size of coords: " << coords.size();
		std::cout << "Size of idx: " << indicies.size();
		for (int i = 0; i < coords.size() / 3; i++) {
			
			float x = coords[3 * i + 0];
			float y = coords[3 * i + 1];
			float z = coords[3 * i + 2];

			out_vert.push_back(make_float4(x, y, z, 0.0f));
		}

		for (int i = 0; i < normals.size() / 3; i++) {

			float n_x = normals[3 * i + 0];
			float n_y = normals[3 * i + 1];
			float n_z = normals[3 * i + 2];

			out_norm.push_back(make_float4(n_x, n_y, n_z, 0.0f));
		}

		for (int i = 0; i < indicies.size(); i++) {
			int idx = indicies[i];

			out_idx.push_back(idx);
		}
	}
}