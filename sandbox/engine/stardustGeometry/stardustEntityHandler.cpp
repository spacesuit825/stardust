#include "stardustEntityHandler.hpp"
#include "stardustEntities.hpp"

#include <fstream>


#define SQR(x) ((x)*(x))

namespace STARDUST {

	float4 getCentreOfMass(std::vector<Sphere>& clump) {
		float4 COM = make_float4(0.0f);

		for (int i = 0; i < clump.size(); i++) {
			COM += clump[i].position;
		}

		return COM / (float)clump.size();
	}

	float4 getCentreOfMass(std::vector<Triangle>& mesh) {
		float4 COM = make_float4(0.0f);

		for (int i = 0; i < mesh.size(); i++) {
			COM += mesh[i].position;
		}

		return COM / (float)mesh.size();
	}

	float4 getCentreOfMass(std::vector<Polyhedron>& polyhedron) {
		float4 COM = make_float4(0.0f);

		for (int i = 0; i < polyhedron.size(); i++) {
			COM += polyhedron[i].position;
		}

		return COM / (float)polyhedron.size();
	}

	float9 computeInertiaTensor(Sphere sphere) {
		float9 I;

		I[0] = 2 / 5 * sphere.mass * SQR(sphere.radius);
		I[4] = 2 / 5 * sphere.mass * SQR(sphere.radius);
		I[8] = 2 / 5 * sphere.mass * SQR(sphere.radius);

		I[1] = I[2] = I[3] = I[5] = I[6] = I[7] = 0.0f;

		return I;
	}

	float9 computeInertiaTensor(Polyhedron polyhedron) {
		float9 I_c = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

		float4 poly_center;

		for (int simplex = 0; simplex < polyhedron.indices.size() / 3; simplex++) {

			float4 a = polyhedron.vertices[polyhedron.indices[3 * simplex + 0]] - polyhedron.position;
			float4 b = polyhedron.vertices[polyhedron.indices[3 * simplex + 1]] - polyhedron.position;
			float4 c = polyhedron.vertices[polyhedron.indices[3 * simplex + 2]] - polyhedron.position;

			float4 d = a + b + c;

			float det = abs(dot(a, make_float4(cross(make_float3(b), make_float3(c)))));

			float4 center = det * d;
			float4 relative_position = center - polyhedron.position;

			I_c[0] += det * (SQR(a.x) + SQR(b.x) + SQR(c.x) + SQR(d.x)) + (SQR(relative_position.y) + SQR(relative_position.z));
			I_c[4] += det * (SQR(a.y) + SQR(b.y) + SQR(c.y) + SQR(d.y)) + (SQR(relative_position.x) + SQR(relative_position.z));
			I_c[8] += det * (SQR(a.z) + SQR(b.z) + SQR(c.z) + SQR(d.z)) + (SQR(relative_position.x) + SQR(relative_position.y));

			I_c[1] += det * (a.x * a.y + b.x * b.y + c.x * c.y + d.x * d.y) + relative_position.x * relative_position.y;
			I_c[2] += det * (a.x * a.z + b.x * b.z + c.x * c.z + d.x * d.z) + relative_position.x * relative_position.z;
			I_c[5] += det * (a.z * a.y + b.z * b.y + c.z * c.y + d.z * d.y) + relative_position.y * relative_position.z;

			I_c[3] += I_c[1];
			I_c[6] += I_c[2];
			I_c[7] += I_c[5];

		}

		float9 I;

		

		I[0] = I_c[4] + I_c[8];
		I[4] = I_c[0] + I_c[8];
		I[8] = I_c[0] + I_c[4];

		I[1] = I[3] = -I_c[1];
		I[2] = I[6] = -I_c[2];
		I[5] = I[7] = -I_c[5];

		std::cout << I[0];

		return I;
	}

	AABB EntityHandler::computeBoundingBox(Sphere& sphere) {
		AABB aabb;

		aabb.init_upper_extent = sphere.position + sphere.radius;
		aabb.init_lower_extent = sphere.position - sphere.radius;

		return aabb;
	}

	AABB EntityHandler::computeBoundingBox(Triangle& triangle) {
		AABB aabb;

		float4 upper_extent = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f);
		float4 lower_extent = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.0f);

		// Iterate through each 3D point
		for (int i = 0; i < triangle.vertices.size(); i++) {

			if (triangle.vertices[i].x < lower_extent.x) lower_extent.x = triangle.vertices[i].x;
			if (triangle.vertices[i].x > upper_extent.x) upper_extent.x = triangle.vertices[i].x;

			if (triangle.vertices[i].y < lower_extent.y) lower_extent.y = triangle.vertices[i].y;
			if (triangle.vertices[i].y > upper_extent.y) upper_extent.y = triangle.vertices[i].y;

			if (triangle.vertices[i].z < lower_extent.z) lower_extent.z = triangle.vertices[i].z;
			if (triangle.vertices[i].z > upper_extent.z) upper_extent.z = triangle.vertices[i].z;

		}

		aabb.init_upper_extent = upper_extent + padding;
		aabb.init_lower_extent = lower_extent - padding;

		printf("tri upper: %.3f, %.3f, %.3f\n", aabb.init_upper_extent.x, aabb.init_upper_extent.y, aabb.init_upper_extent.z);

		return aabb;
	}

	AABB EntityHandler::computeBoundingBox(Polyhedron& polyhedron) {
		AABB aabb;

		float4 upper_extent = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f);
		float4 lower_extent = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.0f);

		// Iterate through each 3D point
		for (int i = 0; i < polyhedron.vertices.size(); i++) {

			if (polyhedron.vertices[i].x < lower_extent.x) lower_extent.x = polyhedron.vertices[i].x;
			if (polyhedron.vertices[i].x > upper_extent.x) upper_extent.x = polyhedron.vertices[i].x;

			if (polyhedron.vertices[i].y < lower_extent.y) lower_extent.y = polyhedron.vertices[i].y;
			if (polyhedron.vertices[i].y > upper_extent.y) upper_extent.y = polyhedron.vertices[i].y;

			if (polyhedron.vertices[i].z < lower_extent.z) lower_extent.z = polyhedron.vertices[i].z;
			if (polyhedron.vertices[i].z > upper_extent.z) upper_extent.z = polyhedron.vertices[i].z;

		}

		aabb.init_upper_extent = upper_extent;
		aabb.init_lower_extent = lower_extent;

		return aabb;
	}
	
	void EntityHandler::addEntity(std::vector<Sphere> clump) {
		
		int entity_idx = entities.size();

		int hull_idx = hulls.size();
		int n_hulls = clump.size();

		float Ixx, Iyy, Izz, Ixy, Ixz, Iyz;
		Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0f;

		Entity entity;

		float entity_mass = 0.0f;

		entity.type = CLUMP;

		entity.primitive_idx = hull_idx;
		entity.n_primitives = n_hulls;

		entity.is_active = true;
		entity.is_visible = true;

		entity.position = getCentreOfMass(clump);
		entity.init_position = entity.position;

		entity.velocity = make_float4(0.0f);
		entity.angular_velocity = make_float4(0.0f);

		entity.linear_momentum = make_float4(0.0f);
		entity.angular_momentum = make_float4(0.0f);

		//entity.force = make_float4(0.0f);
		//entity.torque = make_float4(0.0f);

		entity.quaternion = make_float4(1.0f, 0.0f, 0.0f, 0.0f); // Unit Quaternion

		// Add all the hulls to the hull vector
		for (int prim = 0; prim < clump.size(); prim++) {

			Sphere sphere = clump[prim];

			Hull hull;

			hull.type = SPHERE;

			hull.entity_owner = entity_idx;

			hull.position = sphere.position;
			hull.mass = sphere.mass;
			hull.radius = sphere.radius;

			hull.vertex_idx = -1;
			hull.n_vertices = 0;

			hull.is_active = true;
			hull.is_visible = true;

			hull.force = make_float4(0.0f);

			float4 relative_position = sphere.position - entity.position;

			hull.initial_relative_position = relative_position;
			hull.relative_position = relative_position;

			float9 I = computeInertiaTensor(sphere);

			Ixx += I[0] + sphere.mass * (SQR(relative_position.y) + SQR(relative_position.z));
			Iyy += I[4] + sphere.mass * (SQR(relative_position.x) + SQR(relative_position.z));
			Izz += I[8] + sphere.mass * (SQR(relative_position.x) + SQR(relative_position.y));
			Ixy += I[1] + -sphere.mass * relative_position.x * relative_position.y;
			Ixz += I[2] + -sphere.mass * relative_position.x * relative_position.z;
			Iyz += I[5] + -sphere.mass * relative_position.y * relative_position.z;

			entity_mass += hull.mass;

			hull.normal_stiffness = sphere.normal_stiffness;
			hull.damping = sphere.damping;
			hull.tangential_stiffness = sphere.tangential_stiffness;

			hulls.push_back(hull);
			aabb.push_back(computeBoundingBox(sphere));
		}

		entity.mass = entity_mass;

		if (entity.mass > 10000.0f) {
			entity.is_active = false;
		}

		entity.inertia_tensor = { Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz };

		entities.push_back(entity);

		entity_force.push_back(make_float4(0.0f));
		entity_torque.push_back(make_float4(0.0f));

		n_entities = entity_idx;
	}

	void EntityHandler::addEntity(std::vector <Triangle> mesh) {
		int entity_idx = entities.size();

		int hull_idx = hulls.size();
		int n_hulls = mesh.size();

		float Ixx, Iyy, Izz, Ixy, Ixz, Iyz;
		Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0f;

		Entity entity;

		entity.type = MESH;

		float entity_mass = 0.0f;

		entity.primitive_idx = hull_idx;
		entity.n_primitives = n_hulls;

		entity.is_active = true;
		entity.is_visible = true;

		entity.position = getCentreOfMass(mesh);
		entity.init_position = entity.position;

		entity.velocity = make_float4(0.0f);
		entity.angular_velocity = make_float4(0.0f);

		entity.linear_momentum = make_float4(0.0f);
		entity.angular_momentum = make_float4(0.0f);

		//entity.force = make_float4(0.0f);
		//entity.torque = make_float4(0.0f);

		entity.quaternion = make_float4(1.0f, 0.0f, 0.0f, 0.0f); // Unit Quaternion

		// Add all the hulls to the hull vector
		for (int prim = 0; prim < mesh.size(); prim++) {

			Triangle triangle = mesh[prim];

			Hull hull;

			hull.type = TRIANGLE;

			hull.entity_owner = entity_idx;

			hull.position = triangle.position;
			hull.mass = 1;
			hull.radius = 0.001f; // DONT KNOW ABOUT THIS

			int vertex_idx = vertex.size();

			for (int vert = 0; vert < triangle.vertices.size(); vert++) {
				vertex.push_back(triangle.vertices[vert]);
			}

			hull.vertex_idx = vertex_idx;
			hull.n_vertices = triangle.vertices.size();

			hull.is_active = true;
			hull.is_visible = true;

			hull.force = make_float4(0.0f);

			float4 relative_position = triangle.position - entity.position;

			hull.initial_relative_position = relative_position;
			hull.relative_position = relative_position;

			Ixx += triangle.mass * (SQR(relative_position.y) + SQR(relative_position.z));
			Iyy += triangle.mass * (SQR(relative_position.x) + SQR(relative_position.z));
			Izz += triangle.mass * (SQR(relative_position.x) + SQR(relative_position.y));
			Ixy += -triangle.mass * relative_position.x * relative_position.y;
			Ixz += -triangle.mass * relative_position.x * relative_position.z;
			Iyz += -triangle.mass * relative_position.y * relative_position.z;

			entity_mass += hull.mass;

			hull.normal_stiffness = triangle.normal_stiffness;
			hull.damping = triangle.damping;
			hull.tangential_stiffness = triangle.tangential_stiffness;

			hulls.push_back(hull);
			aabb.push_back(computeBoundingBox(triangle));
		}

		entity.mass = entity_mass;

		entity.inertia_tensor = { Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz };

		entities.push_back(entity);

		entity_force.push_back(make_float4(0.0f));
		entity_torque.push_back(make_float4(0.0f));

		n_entities = entity_idx;

	}

	void EntityHandler::addEntity(std::vector<Polyhedron> complex_polyhedron) {
		int entity_idx = entities.size();

		int hull_idx = hulls.size();
		int n_hulls = complex_polyhedron.size();

		float Ixx, Iyy, Izz, Ixy, Ixz, Iyz;
		Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0f;

		Entity entity;

		entity.type = COMPLEX_POLYHEDRON;

		float entity_mass = 0.0f;

		entity.primitive_idx = hull_idx;
		entity.n_primitives = n_hulls;

		entity.is_active = true;
		entity.is_visible = true;

		entity.position = getCentreOfMass(complex_polyhedron);
		entity.init_position = entity.position;

		entity.velocity = make_float4(0.0f);
		entity.angular_velocity = make_float4(0.0f);

		entity.linear_momentum = make_float4(0.0f);
		entity.angular_momentum = make_float4(0.0f);

		//entity.force = make_float4(0.0f);
		//entity.torque = make_float4(0.0f);

		entity.quaternion = make_float4(1.0f, 0.0f, 0.0f, 0.0f); // Unit Quaternion

		// Add all the hulls to the hull vector
		for (int prim = 0; prim < complex_polyhedron.size(); prim++) {

			Polyhedron polyhedron = complex_polyhedron[prim];

			Hull hull;

			hull.type = POLYHEDRA;

			hull.entity_owner = entity_idx;

			hull.position = polyhedron.position;
			hull.mass = polyhedron.mass;
			hull.radius = -1;

			int vertex_idx = vertex.size();

			for (int vert = 0; vert < polyhedron.vertices.size(); vert++) {
				vertex.push_back(polyhedron.vertices[vert]);
			}

			hull.vertex_idx = vertex_idx;
			hull.n_vertices = polyhedron.vertices.size();

			hull.is_active = true;
			hull.is_visible = true;

			hull.force = make_float4(0.0f);

			float4 relative_position = polyhedron.position - entity.position; // CAREFUL THIS IS FOR TESTING

			hull.initial_relative_position = relative_position;
			hull.relative_position = relative_position;

			float9 I = computeInertiaTensor(polyhedron);

			Ixx += polyhedron.mass * (I[0] + (SQR(relative_position.y) + SQR(relative_position.z)));
			Iyy += polyhedron.mass * (I[4] + (SQR(relative_position.x) + SQR(relative_position.z)));
			Izz += polyhedron.mass * (I[8] + (SQR(relative_position.x) + SQR(relative_position.y)));
			Ixy += polyhedron.mass * (I[1] + -relative_position.x * relative_position.y);
			Ixz += polyhedron.mass * (I[2] + -relative_position.x * relative_position.z);
			Iyz += polyhedron.mass * (I[5] + -relative_position.y * relative_position.z);

			entity_mass += hull.mass;

			hull.normal_stiffness = polyhedron.normal_stiffness;
			hull.damping = polyhedron.damping;
			hull.tangential_stiffness = polyhedron.tangential_stiffness;

			hulls.push_back(hull);
			aabb.push_back(computeBoundingBox(polyhedron));
		}

		entity.mass = entity_mass;

		entity.inertia_tensor = { Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz };

		entities.push_back(entity);

		entity_force.push_back(make_float4(0.0f));
		entity_torque.push_back(make_float4(0.0f));
		n_entities = entity_idx;
	}

	void EntityHandler::allocate() {

		d_hull = hulls;
		d_entity = entities;
		d_init_vertex = vertex;
		d_vertex = vertex;
		d_aabb = aabb;
		d_entity_force = entity_force;
		d_entity_torque = entity_torque;

		d_hull_ptr = thrust::raw_pointer_cast(d_hull.data());
		d_entity_ptr = thrust::raw_pointer_cast(d_entity.data());
		d_init_vertex_ptr = thrust::raw_pointer_cast(d_init_vertex.data());
		d_vertex_ptr = thrust::raw_pointer_cast(d_vertex.data());
		d_aabb_ptr = thrust::raw_pointer_cast(d_aabb.data());
		d_entity_force_ptr = thrust::raw_pointer_cast(d_entity_force.data());
		d_entity_torque_ptr = thrust::raw_pointer_cast(d_entity_torque.data());

		device_geometry_data.d_hull_ptr = d_hull_ptr;
		device_geometry_data.d_entity_ptr = d_entity_ptr;
		device_geometry_data.d_init_vertex_ptr = d_init_vertex_ptr;
		device_geometry_data.d_vertex_ptr = d_vertex_ptr;
		device_geometry_data.d_aabb_ptr = d_aabb_ptr;
		device_geometry_data.d_entity_force_ptr = d_entity_force_ptr;
		device_geometry_data.d_entity_torque_ptr = d_entity_torque_ptr;

	}

	void EntityHandler::writeToVTK(int time_step) {
		hulls = d_hull;
		vertex = d_vertex;

		std::vector<float4> spheres_ls;
		std::vector<int> indicies_ls;
		std::vector<float4> vertices_ls;

		for (int i = 0; i < hulls.size(); i++) {
			if (hulls[i].type == SPHERE) {
				spheres_ls.push_back(hulls[i].position);
			}
			else if (hulls[i].type == TRIANGLE) {
				//printf("huh1 %d\n", hulls[i].n_vertices);
				for (int j = 0; j < hulls[i].n_vertices; j++) {

					//printf("huh\n");
					vertices_ls.push_back(vertex[hulls[i].vertex_idx + j]);

				}

				indicies_ls.push_back(i);
			}
			else if (hulls[i].type == POLYHEDRA) {
				for (int j = 0; j < hulls[i].n_vertices; j++) {

					//printf("huh\n");
					vertices_ls.push_back(vertex[hulls[i].vertex_idx + j]);

				}

				indicies_ls.push_back(i);
			}
		}

		std::ofstream vtkFile("C:/Users/lachl/Documents/stardust/sandbox/vtkFiles/out_" + std::to_string(time_step) + ".vtk");

		// Write VTK header
		vtkFile << "# vtk DataFile Version 2.0" << std::endl;
		vtkFile << "VTK file for your point cloud data" << std::endl;
		vtkFile << "ASCII" << std::endl;
		vtkFile << "DATASET POLYDATA" << std::endl;

		vtkFile << "POINTS " << spheres_ls.size() << " float" << std::endl;

		for (size_t i = 0; i < spheres_ls.size(); ++i) {
			vtkFile << spheres_ls[i].x << " " << spheres_ls[i].y << " " << spheres_ls[i].z << std::endl;
		}

		// Write point data (scalar values)
		vtkFile << "POINT_DATA " << spheres_ls.size() << std::endl;
		vtkFile << "SCALARS radius float 1" << std::endl;
		vtkFile << "LOOKUP_TABLE default" << std::endl;
		for (size_t i = 0; i < spheres_ls.size(); ++i) {
			vtkFile << 0.5f << std::endl;
		}

		vtkFile.close();


		std::ofstream vtk2File("C:/Users/lachl/Documents/stardust/sandbox/vtkFiles/triout_" + std::to_string(time_step) + ".vtk");

		//// Write VTK header
		vtk2File << "# vtk DataFile Version 2.0" << std::endl;
		vtk2File << "VTK file for your point cloud data" << std::endl;
		vtk2File << "ASCII" << std::endl;
		vtk2File << "DATASET POLYDATA" << std::endl;

		vtk2File << "POINTS " << vertices_ls.size() << " float" << std::endl;

		for (size_t i = 0; i < vertices_ls.size(); ++i) {
			vtk2File << vertices_ls[i].x << " " << vertices_ls[i].y << " " << vertices_ls[i].z << std::endl;
		}

		vtk2File << "TRIANGLE_STRIPS " << indicies_ls.size() << " " << indicies_ls.size() * 4 << std::endl;
		for (int i = 0; i < indicies_ls.size(); i++) {
			vtk2File << "3 " << 3 * i + 0 << " " << 3 * i + 1 << " " << 3 * i + 2 << std::endl;
		}

		vtk2File.close();

		//std::ofstream vtk3File("C:/Users/lachl/Documents/stardust/sandbox/vtkFiles/polyout_" + std::to_string(time_step) + ".vtk");

		//// Write VTK header
		//vtk3File << "# vtk DataFile Version 2.0" << std::endl;
		//vtk3File << "VTK file for your point cloud data" << std::endl;
		//vtk3File << "ASCII" << std::endl;
		//vtk3File << "DATASET POLYDATA" << std::endl;

		//vtk3File << "POINTS " << vertices_ls.size() << " float" << std::endl;

		//for (size_t i = 0; i < vertices_ls.size(); ++i) {
		//	vtk3File << vertices_ls[i].x << " " << vertices_ls[i].y << " " << vertices_ls[i].z << std::endl;
		//}

		//vtk3File << "TRIANGLE_STRIPS " << 4 << " " << 4 * 4 << std::endl;
		///*for (int i = 0; i < indicies_ls.size(); i++) {
		//	vtk3File << "3 " << 3 * i + 0 << " " << 3 * i + 1 << " " << 3 * i + 2 << std::endl;
		//}*/
		//vtk3File << "3 " << 0 << " " << 1 << " " << 3 << std::endl;
		//vtk3File << "3 " << 1 << " " << 2 << " " << 3 << std::endl;
		//vtk3File << "3 " << 0 << " " << 2 << " " << 3 << std::endl;
		//vtk3File << "3 " << 0 << " " << 2 << " " << 1 << std::endl;

		//vtk3File.close();

		//std::cout << "Write success!\n";
	}

}