#ifndef RL_NET_H
#define RL_NET_H



#include <torch/torch.h>

namespace ns3 {

struct FeatureVector {
  FeatureVector();

  // Implement the Net's algorithm.
  torch::Tensor getCurrentVector();
  void updateNewPacket(double interarrivalTime, uint64_t queueSize);
  void setBw(double bw);
  void setDelay(double delay);
  void updateNewLoss(uint8_t drop);
  void updateQueue(uint32_t queue_size_arg);
  void updateNewDeparture(double interdepartureTime);

  std::vector<double> invert(std::vector<double> v);
  std::vector<double> sqrt_vector(std::vector<double> v);
  std::vector<double> skewness_vector(std::vector<double> v, std::vector<double> means, std::vector<double> stds);

  static const uint64_t queueVectorLength = 10;
  static const uint64_t queueMaxVectorLength = 10;
  static const uint64_t arrivalVectorLength = 10;
  static const uint64_t departureVectorLength = 10;
  static const uint64_t dropVectorLength = 10;

  uint64_t already_updated_arrival = 0;
  uint64_t already_updated_departure = 0;
  double last_loss = 0.0;
  double m_bw;
  double m_delay;

  std::vector<double> queueVector;
  std::vector<double> queueVectorStd;
  std::vector<double> queueVectorToThePowerOfThree;
  std::vector<double> queueMaxVector;
  std::vector<double> arrivalVector;
  std::vector<double> departureVector;
  std::vector<double> dropVector;
  std::vector<double> dropVectorDiff;
};

struct RlNet : torch::nn::Module {
  RlNet();

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
};

struct RlNetReward : torch::nn::Module {
  RlNetReward();

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear fc1_reward{nullptr}, fc2_reward{nullptr}, fc3_reward{nullptr}, fc4_reward{nullptr};
};

}

#endif