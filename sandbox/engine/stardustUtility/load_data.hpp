#ifndef _STARDUST_PRELIM_DATA_LOADER_HEADER_
#define _STARDUST_PRELIM_DATA_LOADER_HEADER_

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "../stardustGeometry/stardustPrimitives.hpp"

union Prim {
    Sphere sphere;
    Triangle triangle;
    Polyhedron polyhedron;
}

struct EntityData {
    int entity_type;
    int n_primitives;

    int spawn_type;
    int spawn_centrality;
    float4 position;
    AABB spawn_bounds;

    std::string entity_name;

    std::vector<Prim>> primitives;
}

void loadPrimitiveData(std::string file_name, std::vector<std::vector<Prim>> entity_data) {

    std::string line;
    std::ifstream myfile(file_name);
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            std::cout << line << '\n';
        }
        myfile.close();
    }

    else std::cout << "Unable to open file";









}







#endif // _STARDUST_PRELIM_DATA_LOADER_HEADER_