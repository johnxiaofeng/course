#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <vector>

struct Particle
{
	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};

using Particles = std::vector<Particle>;

#endif // PARTICLE_H_
