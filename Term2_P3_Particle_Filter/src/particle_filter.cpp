/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <algorithm>
#include <cassert>
#include <random>
#include <math.h>
#include <numeric>

#include "math_utils.h"

ParticleFilter::ParticleFilter()
: m_numParticles(1000)
, m_initialized(false)
{

}

ParticleFilter::~ParticleFilter()
{

}

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	assert(!m_initialized);

	const double stdX = std[0];
	std::normal_distribution<double> distributionX(x, stdX);

	const double stdY = std[1];
	std::normal_distribution<double> distributionY(y, stdY);

	const double stdTheta = std[2];
	std::normal_distribution<double> distributionTheta(theta, stdTheta);

	m_particles.reserve(m_numParticles);
	std::default_random_engine generator;
	for (int index = 0; index < m_numParticles; index++)
	{
		Particle particle;
		particle.id = index;
		particle.weight = 1.0;
		particle.x = distributionX(generator);
		particle.y = distributionY(generator);
		particle.theta = distributionTheta(generator);

		m_particles.push_back(particle);
	}

	m_initialized = true;
}

// TODO: Add measurements to each particle and add random Gaussian noise.
// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
//  http://www.cplusplus.com/reference/random/default_random_engine/
void ParticleFilter::prediction(double deltaTime, double std[], double velocity, double yawRate)
{
	// distributions for the random noise
	std::default_random_engine generator;

	const double stdX = std[0];
	std::normal_distribution<double> distributionX(0, stdX);

	const double stdY = std[1];
	std::normal_distribution<double> distributionY(0, stdY);

	const double stdTheta = std[2];
	std::normal_distribution<double> distributionTheta(0, stdTheta);

	// predict based on velocity and yawRate
	if (MathUtils::DoubleEqual(yawRate, 0.0))
	{
		for (Particle& particle : m_particles)
		{
			particle.x = particle.x + velocity * deltaTime * cos(particle.theta) + distributionX(generator);
			particle.y = particle.y + velocity * deltaTime * sin(particle.theta) + distributionY(generator);
			particle.theta = particle.theta + distributionTheta(generator);
		}
	}
	else
	{
		for (Particle& particle : m_particles)
		{
			particle.x = particle.x + (velocity * (sin(particle.theta + yawRate * deltaTime) - sin(particle.theta))) / yawRate + distributionX(generator);
			particle.y = particle.y + (velocity * (cos(particle.theta) - cos(particle.theta + yawRate * deltaTime))) / yawRate + distributionY(generator);
			particle.theta = particle.theta + yawRate * deltaTime + distributionTheta(generator);
		}
	}
}

// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
//   according to the MAP'S coordinate system. You will need to transform between the two systems.
//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
//   The following is a good resource for the theory:
//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//   and the following is a good resource for the actual equation to implement (look at equation
//   3.33
//   http://planning.cs.uiuc.edu/node99.html
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	for (Particle& particle : m_particles)
	{
		double newWeight = 1.0;

		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();

		for (const LandmarkObs& observation : observations)
		{
			// transform observation to map coordinate
			const double mapX = particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
			const double mapY = particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);

			// associate the observation to landmark
			Map::single_landmark_s associatedLandmark;
			if (!associateLandmarkWithObservationInMap(map_landmarks, sensor_range, mapX, mapY, associatedLandmark))
			{
				assert(false);
				continue;
			}

			// calculate the weight of particle to associated landmark
			const double stdX = std_landmark[0];
			const double stdY = std_landmark[1];
			const double diffX = mapX - associatedLandmark.x_f;
			const double diffY = mapY - associatedLandmark.y_f;
			const double weight = (1.0 / (2 * M_PI * stdX * stdY)) * exp(-((diffX * diffX ) / (2 * stdX * stdX) + (diffY * diffY) / (2 * stdY * stdY)));
			newWeight *= weight;

			particle.associations.push_back(associatedLandmark.id_i);
			particle.sense_x.push_back(mapX);
			particle.sense_y.push_back(mapY);
		}

		particle.weight = newWeight;
	}
}

bool ParticleFilter::associateLandmarkWithObservationInMap(const Map& map, double sensorRange, double obsXInMap, double obsYInMap, Map::single_landmark_s& associatedLandmark)
{
	int currentIndex = -1;
	double currentDistance = sensorRange;

	const int numLandmarks = static_cast<int>(map.landmark_list.size());
	for (int index = 0; index < numLandmarks; index++)
	{
		const double distance = dist(obsXInMap, obsYInMap, map.landmark_list[index].x_f, map.landmark_list[index].y_f);
		if (distance < currentDistance)
		{
			currentDistance = distance;
			currentIndex = index;
		}
	}

	if (-1 == currentIndex)
	{
		return false;
	}

	associatedLandmark = map.landmark_list[currentIndex];
	return true;
}

void ParticleFilter::resample()
{
	// create the vector for the resampled particles
	Particles resampledParticles;
	resampledParticles.reserve(m_numParticles);

	// get the weights of all particles to be used by the discrete distribution.
	std::vector<double> weights;
	getWeights(weights);

	// do the resampling using the discrete distribution provided by c++11
	std::default_random_engine generator;
	std::discrete_distribution<> discreteDistribution(weights.begin(), weights.end());
	for (size_t index = 0; index < m_numParticles; index++)
	{
		int particleIndex = discreteDistribution(generator);
		resampledParticles.push_back(m_particles[particleIndex]);
	}

	// use the resampled particles as the particles
	m_particles.swap(resampledParticles);
}

void ParticleFilter::resampleFast()
{
	// create the vector for the resampled particles
	Particles resampledParticles;
	resampledParticles.reserve(m_numParticles);

	// use the undiform distribution [0,1) for resampling
	std::default_random_engine generator;
	std::uniform_real_distribution<double> uniformDistribution;

	// do the resampling using the method taught by Sebastian
	const double maxWeight = getMaxWeight();
	int index = static_cast<int>(uniformDistribution(generator) * m_numParticles);
	double beta = 0.0;
	for (int particleIndex = 0;  particleIndex < m_numParticles; particleIndex++)
	{
		beta += uniformDistribution(generator) * 2.0 * maxWeight;
		while (beta > m_particles[index].weight)
		{
			beta -= m_particles[index].weight;
			index = (index + 1) % m_numParticles;
		}
		resampledParticles.push_back(m_particles[index]);
	}

	// use the resampled particles as the particles
	m_particles.swap(resampledParticles);
}

const Particles& ParticleFilter::getParticles() const
{
	return m_particles;
}

bool ParticleFilter::initialized() const
{
	return m_initialized;
}

std::string ParticleFilter::getAssociations(const Particle& best) const
{
	return getStringFromDataVector(best.associations);
}

std::string ParticleFilter::getSenseX(const Particle& best) const
{
	return getStringFromDataVector(best.sense_x);
}

std::string ParticleFilter::getSenseY(const Particle& best) const
{
	return getStringFromDataVector(best.sense_y);
}

double ParticleFilter::getMaxWeight() const
{
	double maxWeight = 0;
	std::for_each(m_particles.begin(), m_particles.end(), [&maxWeight](const Particle& particle)
	{
		if (particle.weight > maxWeight)
		{
			maxWeight = particle.weight;
		}
	});

	return maxWeight;
}

void ParticleFilter::getWeights(std::vector<double>& weights) const
{
	weights.clear();
	weights.reserve(m_particles.size());
	for (const Particle& particle : m_particles)
	{
		weights.push_back(particle.weight);
	}
}