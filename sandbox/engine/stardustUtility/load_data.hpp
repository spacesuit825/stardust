#ifndef _STARDUST_PRELIM_DATA_LOADER_HEADER_
#define _STARDUST_PRELIM_DATA_LOADER_HEADER_

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "../../external/stl_reader/stl_reader.h"

#include "../stardustGeometry/stardustPrimitives.hpp"

// union Prim {
//     Sphere sphere;
//     Triangle triangle;
//     Polyhedron polyhedron;
// }

// struct EntityData {
//     int entity_type;
//     int n_primitives;

//     int spawn_type;
//     int spawn_centrality;
//     float4 position;
//     AABB spawn_bounds;

//     std::string entity_name;

//     std::vector<Prim>> primitives;
// }

// void loadPrimitiveData(std::string file_name, std::vector<std::vector<Prim>> entity_data) {

//     std::string line;
//     std::ifstream myfile(file_name);
//     if (myfile.is_open())
//     {
//         while (getline(myfile, line))
//         {
//             std::cout << line << '\n';
//         }
//         myfile.close();
//     }

//     else std::cout << "Unable to open file";
// }

void load_stl(const char* filepath, std::vector<STARDUST::Triangle>& triangles) {

    std::vector<float> coords, normals;
    std::vector<unsigned int> tris, solids;

    try {
        stl_reader::ReadStlFile (filepath, coords, normals, tris, solids);
        const size_t numTris = tris.size() / 3;
        for(size_t itri = 0; itri < numTris; ++itri) {
            std::cout << "coordinates of triangle " << itri << ": ";

            STARDUST::Triangle triangle;

            for(size_t icorner = 0; icorner < 3; ++icorner) {
                float* c = &coords[3 * tris [3 * itri + icorner]];
                std::cout << "(" << c[0] << ", " << c[1] << ", " << c[2] << ") ";

                float4 vertex = make_float4(c[0], c[1], c[2], 0.0f);
                triangle.vertices.push_back(vertex);
            }
            std::cout << std::endl;
            
            float4 position = make_float4(0.0f);
            for (int i = 0; i < 3; i++) {
                position += triangle.vertices[i];
            }

            triangle.position = 1/3 * position;
            triangle.mass = 1.0f;

            triangle.normal_stiffness = 1e+07;
	        triangle.damping = 0.02f;
	        triangle.tangential_stiffness = 1.0f;

        
            float* n = &normals [3 * itri];
            std::cout   << "normal of triangle " << itri << ": "
                        << "(" << n[0] << ", " << n[1] << ", " << n[2] << ")\n";

            triangles.push_back(triangle);
        }
    }
        catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }





}







#endif // _STARDUST_PRELIM_DATA_LOADER_HEADER_