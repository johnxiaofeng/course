/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"
#include "particle.h"

class ParticleFilter
{
public:
	ParticleFilter();
	~ParticleFilter();

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the
	 *   observed measurements.
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map& map_landmarks);

	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 * using the simplest way with cpp discrete distribution
	 */
	void resample();

	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 * using the method mentioned by Sebastian
	 */
	void resampleFast();

	/**
	 * get the associate of the particle for debugging purpose
	 */
	std::string getAssociations(const Particle& best) const;

	/**
	 * get the measurement x in map coordinate of the particle for debuging purpose
	 */
	std::string getSenseX(const Particle& best) const;

	/**
	 * get the measurement y in map coordinate of the particle for debuging purpose
	 */
	std::string getSenseY(const Particle& best) const;

	/**
	 * get all the particles in the current particle filter
	 */
	const Particles& getParticles() const;

	/**
	* initialized Returns whether particle filter is initialized yet or not.
	*/
	bool initialized() const;

private:
	/**
	 * get the maximum weight from all particles
	 */
	double getMaxWeight() const;

	/**
	 * get the weight of all particles and put them into the weights vector
	 */
	void getWeights(std::vector<double>& weights) const;

	/**
	 * Associate a landmark to an observation given in
	 * return false, if there is no landmark to associate
	 * return true, if succeed in finding the landmark to associate
	 */
	bool associateLandmarkWithObservationInMap(
		const Map&map,
		double sensorRange,
		double obsXInMap,
		double obsYInMap,
		Map::single_landmark_s& landmark);

private:
	// Number of particles to draw
	const int m_numParticles;

	// Set of current particles
	std::vector<Particle> m_particles;

	// Flag, if filter is initialized
	bool m_initialized;
};


#endif /* PARTICLE_FILTER_H_ */
