#include "rl-net.h"
#include "ns3/core-module.h"
#include <cmath>

double queue_normalization_factor = 100;
double speed_normalization_factor = 10000;
double loss_normalizer = 10;

uint32_t WIDTH = 256;
double initialCurrentValue = ((double) 1)/16;

namespace ns3 {

FeatureVector::FeatureVector() : queueVector(queueVectorLength, 0), queueVectorStd(queueVectorLength, 0), queueVectorToThePowerOfThree(queueVectorLength, 0), queueMaxVector(queueMaxVectorLength, 0), arrivalVector(arrivalVectorLength, 0), departureVector(departureVectorLength, 0), dropVector(dropVectorLength, 0) {}

void FeatureVector::updateNewPacket(double interarrivalTime, uint64_t queueSize) {
	already_updated_arrival += 1;

	double queueSizeDouble = (double) queueSize;
	queueSizeDouble /= queue_normalization_factor;
	double currentValue = initialCurrentValue;
	for (size_t i = 0; i < queueVectorLength; i++) {
		currentValue /= 2;
		queueVector[i] = currentValue*queueSizeDouble + (1-currentValue)*queueVector[i];
	}

	currentValue = initialCurrentValue;
	for (size_t i = 0; i < queueVectorLength; i++) {
		currentValue /= 2;
		double diff = queueVector[i] - queueSizeDouble;
		queueVectorStd[i] = currentValue*diff*diff + (1-currentValue)*queueVectorStd[i];
	}

	// currentValue = initialCurrentValue;
	// for (size_t i = 0; i < queueVectorLength; i++) {
	// 	currentValue /= 2;
	// 	queueVectorToThePowerOfThree[i] = currentValue*queueSizeDouble*queueSizeDouble*queueSizeDouble + (1-currentValue)*queueVectorToThePowerOfThree[i];
	// }

	currentValue = initialCurrentValue;
	for (size_t i = 0; i < arrivalVectorLength; i++) {
		currentValue /= 2;
		arrivalVector[i] = currentValue*interarrivalTime + (1-currentValue)*arrivalVector[i];
	}

}

void FeatureVector::updateNewLoss(uint8_t drop) {
	auto now = (double) Simulator::Now().GetSeconds();
	if (drop == 1) {
		last_loss = now;
	}
	double currentValue = initialCurrentValue;
	for (size_t i = 0; i < dropVectorLength; i++) {
		currentValue /= 2;
		dropVector[i] = currentValue*(now-last_loss)/loss_normalizer + (1-currentValue)*dropVector[i];
	}
}

void FeatureVector::setBw(double bw) {
	m_bw = (double) bw;
}

void FeatureVector::setDelay(double delay) {
	m_delay = (double) delay;
}

void FeatureVector::updateQueue(uint32_t queue_size_arg) {

	double queueSizeDouble = (double) queue_size_arg;
	queueSizeDouble /= queue_normalization_factor;
	// auto queueSizeDiff = queueSizeDouble - queueVector[0];
	double currentValue = initialCurrentValue;
	for (size_t i = 0; i < queueMaxVectorLength; i++) {
		currentValue /= 2;
		// queueVector[i] = queueVector[i]!=0.0 ? currentValue*queueSizeDouble + (1-currentValue)*queueVector[i] : queueSizeDouble;
		queueMaxVector[i] = currentValue*queueSizeDouble + (1-currentValue)*queueMaxVector[i];
	}
}


void FeatureVector::updateNewDeparture(double interdepartureTime) {
	already_updated_departure += 1;
	// auto interdepartureTimeDiff = interdepartureTime - departureVector[0];
	double currentValue = initialCurrentValue;
	for (size_t i = 0; i < departureVectorLength; i++) {
		currentValue /= 2;
		departureVector[i] = currentValue*interdepartureTime + (1-currentValue)*departureVector[i];
	}
	// currentValue = 2;
	// for (size_t i = 0; i < departureVectorLengthDiff; i++) {
	// 	currentValue /= 2;
	// 	departureVectorDiff[i] = currentValue*interdepartureTimeDiff + (1-currentValue)*departureVectorDiff[i];
	// }
}

std::vector<double> FeatureVector::invert(std::vector<double> v) {
	std::vector<double> inverted;
	inverted.reserve(v.size());
	double currentValue = initialCurrentValue;
	for (size_t i = 0; i < v.size(); i++) {
		currentValue /= 2;
		double new_item = (v[i] == 0.0 || 1.0/currentValue > std::min(already_updated_arrival, already_updated_departure) ) ? 0.0 : 1.0/v[i]/speed_normalization_factor;
		inverted.push_back(new_item);
	}
	return inverted;
}

std::vector<double> FeatureVector::sqrt_vector(std::vector<double> v) {
	std::vector<double> sqrted;
	sqrted.reserve(v.size());
	for (size_t i = 0; i < v.size(); i++) {
		double new_item = sqrt(v[i]);
		sqrted.push_back(new_item);
	}
	return sqrted;
}

torch::Tensor FeatureVector::getCurrentVector() {
	auto options_double = torch::TensorOptions().dtype(torch::kFloat64);

	std::vector<double> cattedVector;
	cattedVector.insert(cattedVector.end(),queueVector.begin(),queueVector.end());
	auto queueSizeStdSqrted = sqrt_vector(queueVectorStd);
	cattedVector.insert(cattedVector.end(),queueSizeStdSqrted.begin(),queueSizeStdSqrted.end());
	cattedVector.insert(cattedVector.end(),queueMaxVector.begin(),queueMaxVector.end());
	auto arrivalVectorInverted = invert(arrivalVector);
	cattedVector.insert(cattedVector.end(),arrivalVectorInverted.begin(),arrivalVectorInverted.end());
	auto departureVectorInverted = invert(departureVector);
	cattedVector.insert(cattedVector.end(),departureVectorInverted.begin(),departureVectorInverted.end());
	cattedVector.insert(cattedVector.end(),dropVector.begin(),dropVector.end());

	return torch::tensor(cattedVector, options_double);
}



RlNet::RlNet() {
	fc1 = register_module("fc1", torch::nn::Linear(FeatureVector::queueVectorLength*2+FeatureVector::queueMaxVectorLength+FeatureVector::arrivalVectorLength+FeatureVector::departureVectorLength+FeatureVector::dropVectorLength, WIDTH));
	fc2 = register_module("fc2", torch::nn::Linear(WIDTH, WIDTH));
	fc3 = register_module("fc3", torch::nn::Linear(WIDTH, WIDTH));
	fc4 = register_module("fc4", torch::nn::Linear(WIDTH, 1));
}

// Implement the Net's algorithm.
torch::Tensor RlNet::forward(torch::Tensor x) {
	// std::cout << "x " << x << std::endl;
	// std::cout << "fc1 " << fc1 << std::endl;
	x = torch::leaky_relu(fc1->forward(x));
	x = torch::leaky_relu(fc2->forward(x));
	x = torch::leaky_relu(fc3->forward(x));
	x = fc4->forward(x);
	return x;
}

RlNetReward::RlNetReward() {
	fc1_reward = register_module("fc1_reward", torch::nn::Linear(FeatureVector::queueVectorLength*2+FeatureVector::queueMaxVectorLength+FeatureVector::arrivalVectorLength+FeatureVector::departureVectorLength+FeatureVector::dropVectorLength, WIDTH));
	fc2_reward = register_module("fc2_reward", torch::nn::Linear(WIDTH, WIDTH));
	fc3_reward = register_module("fc3_reward", torch::nn::Linear(WIDTH, WIDTH));
	fc4_reward = register_module("fc4_reward", torch::nn::Linear(WIDTH, 1));
}

// Implement the Net's algorithm.
torch::Tensor RlNetReward::forward(torch::Tensor x) {
	// std::cout << "x " << x << std::endl;
	// std::cout << "fc1 " << fc1 << std::endl;
	x = torch::leaky_relu(fc1_reward->forward(x));
	x = torch::leaky_relu(fc2_reward->forward(x));
	x = torch::leaky_relu(fc3_reward->forward(x));
	x = fc4_reward->forward(x);
	return x;
}

}