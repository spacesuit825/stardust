#include "stardustEntityHandler.hpp"
#include "stardustEntities.hpp"

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

		entity.velocity = make_float4(0.0f);
		entity.angular_velocity = make_float4(0.0f);

		entity.linear_momentum = make_float4(0.0f);
		entity.angular_momentum = make_float4(0.0f);

		entity.force = make_float4(0.0f);
		entity.torque = make_float4(0.0f);

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

			Ixx += sphere.mass * (SQR(relative_position.y) + SQR(relative_position.z));
			Iyy += sphere.mass * (SQR(relative_position.x) + SQR(relative_position.z));
			Izz += sphere.mass * (SQR(relative_position.x) + SQR(relative_position.y));
			Ixy += -sphere.mass * relative_position.x * relative_position.y;
			Ixz += -sphere.mass * relative_position.x * relative_position.z;
			Iyz += -sphere.mass * relative_position.y * relative_position.z;

			entity_mass += hull.mass;

			hull.normal_stiffness = sphere.normal_stiffness;
			hull.damping = sphere.damping;
			hull.tangential_stiffness = sphere.tangential_stiffness;

			hulls.push_back(hull);
			aabb.push_back(computeBoundingBox(sphere));
		}

		entity.mass = entity_mass;

		entity.inertia_tensor = { Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz };

		entities.push_back(entity);
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

		entity.primitive_idx = hull_idx;
		entity.n_primitives = n_hulls;

		entity.is_active = true;
		entity.is_visible = true;

		entity.position = getCentreOfMass(mesh);

		entity.velocity = make_float4(0.0f);
		entity.angular_velocity = make_float4(0.0f);

		entity.linear_momentum = make_float4(0.0f);
		entity.angular_momentum = make_float4(0.0f);

		entity.force = make_float4(0.0f);
		entity.torque = make_float4(0.0f);

		entity.quaternion = make_float4(1.0f, 0.0f, 0.0f, 0.0f); // Unit Quaternion

		// Add all the hulls to the hull vector
		for (int prim = 0; prim < mesh.size(); prim++) {

			Triangle triangle = mesh[prim];

			Hull hull;

			hull.type = TRIANGLE;

			hull.entity_owner = entity_idx;

			hull.position = triangle.position;
			hull.mass = 1;
			hull.radius = -1;

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

			entity.mass += hull.mass;

			hull.normal_stiffness = triangle.normal_stiffness;
			hull.damping = triangle.damping;
			hull.tangential_stiffness = triangle.tangential_stiffness;

			hulls.push_back(hull);
			aabb.push_back(computeBoundingBox(triangle));
		}

		entity.inertia_tensor = { Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz };

		entities.push_back(entity);
		n_entities = entity_idx;

	}

	void EntityHandler::addEntity(std::vector<Polyhedron> complex_polyhedron) {

	}

	void EntityHandler::allocate() {

		d_hull = hulls;
		d_entity = entities;
		d_vertex = vertex;
		d_aabb = aabb;

		d_hull_ptr = thrust::raw_pointer_cast(d_hull.data());
		d_entity_ptr = thrust::raw_pointer_cast(d_entity.data());
		d_vertex_ptr = thrust::raw_pointer_cast(d_vertex.data());
		d_aabb_ptr = thrust::raw_pointer_cast(d_aabb.data());

		device_geometry_data.d_hull_ptr = d_hull_ptr;
		device_geometry_data.d_entity_ptr = d_entity_ptr;
		device_geometry_data.d_vertex_ptr = d_vertex_ptr;
		device_geometry_data.d_aabb_ptr = d_aabb_ptr;

	}

}