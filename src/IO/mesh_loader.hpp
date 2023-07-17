#ifndef _STARDUST_GENERIC_MESH_
#define _STARDUST_GENERIC_MESH_

#include <stl_reader.h>
#include <vector_types.h>
#include "cuda.h"
#include "cuda_runtime.h"

namespace STARDUST {

	void loadAndConvertMesh(const char*, std::vector<float4>&, std::vector<int>&, std::vector<float4>&);

}

#endif // _STARDUST_GENERIC_MESH_