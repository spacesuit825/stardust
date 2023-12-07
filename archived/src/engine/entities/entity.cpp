#include "entity.hpp"
#include "types.hpp"
#include "helper_math.hpp"

#define SQR(x) ((x) * (x))

namespace STARDUST {
	
	void DEMParticle::initParticles(int grid_resolution) {
		// For now we only deal with a cube
		Scalar length = m_size; // m

		// Cast rays in the future
		int num_particles = SQR(floor(length / diameter));
		float dx = length / num_particles;

		// 0, 0, 0 is upper left corner of the cube
		
		for (int i = 1; i <= grid_resolution; i++) {
			for (int j = 1; j <= grid_resolution; j++) {
				for (int k = 1; k <= grid_resolution; k++) {

					DEMSphere particle;

					particle.position = make_float4(i * dx, j * dx, k * dx, 0.0f);
					particle.size = diameter;
					particle.mass = mass / num_particles; // Assume equal mass distribution
					particles.push_back(particle);

				}
			}
		}
	}

	void DEMParticle::getCenterOfMass() {
		for (int i = 0; i < particles.size(); i++) {
			COM += particles[i].position;
		}

		COM = COM / (float)particles.size();
	}

	void DEMParticle::setParticlesInWorldSpace() {

		for (int i = 0; i < particles.size(); i++) {

			DEMSphere& particle = particles[i];
			
			float4 pos = particle.position;

			// Shift reference location to the COM
			float4 COM_pos = pos - COM;

			// Transmute position into world space, no rotation here so simple addition will work
			float4 world_pos = position - COM_pos;
			
			particle.position = world_pos;
		}
	}

	Scalar* DEMParticle::convertParticlePositionsToNakedArray() {

		Scalar* h_position_ptr = new Scalar[3 * particles.size() * sizeof(Scalar)];

		for (int i = 0; i < particles.size(); i++) {

			/*h_position_ptr[3 * i + 0] = particles[i].position(0);
			h_position_ptr[3 * i + 1] = particles[i].position(1);
			h_position_ptr[3 * i + 2] = particles[i].position(2);*/
		}

		return h_position_ptr;
	}

	void DEMMesh::computeBoundingSpheres() {
		
		std::cout << "Num triangle: " << n_triangles << "\n";
		std::cout << "Num vert: " << vertices.size() << "\n";
		std::cout << "Num idx: " << indicies.size() << "\n";
		for (int v = 0; v < n_triangles; v++) {

			int i1 = indicies[3 * v + 0];
			int i2 = indicies[3 * v + 1];
			int i3 = indicies[3 * v + 2];

			float3 p1 = make_float3(vertices[i1].x, vertices[i1].y, vertices[i1].z);
			float3 p2 = make_float3(vertices[i2].x, vertices[i2].y, vertices[i2].z);
			float3 p3 = make_float3(vertices[i3].x, vertices[i3].y, vertices[i3].z);

			float radius;
			float3 position;

			// Calculate relative distances
			float A = length(p1 - p2);
			float B = length(p2 - p3);
			float C = length(p3 - p1);

			// Re-orient triangle (make A longest side)
			const float3* a = &p3, * b = &p1, * c = &p2;
			if (B < C) std::swap(B, C), std::swap(b, c);
			if (A < B) std::swap(A, B), std::swap(a, b);

			// If obtuse, just use longest diameter, otherwise circumscribe
			if ((B * B) + (C * C) <= (A * A)) {
				radius = A / 2.f;
				position = (*b + *c) / 2.f;
			}
			else {
				// http://en.wikipedia.org/wiki/Circumscribed_circle
				float cos_a = (B * B + C * C - A * A) / (B * C * 2);
				radius = A / (sqrt(1 - cos_a * cos_a) * 2.f);
				float3 alpha = *a - *c, beta = *b - *c;
				position = (beta * dot(alpha, alpha) - cross(alpha * dot(beta, beta), cross(alpha, beta))) /
					(dot(cross(alpha, beta), cross(alpha, beta)) * 2.f) + *c;
			}

			positions.push_back(make_float4(position.x, position.y, position.z, 0.0f));
			radii.push_back(radius);
		}
	}
}