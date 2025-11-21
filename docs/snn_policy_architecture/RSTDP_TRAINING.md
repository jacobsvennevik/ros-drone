# R-STDP Training Guide

## Overview

R-STDP (Reward-modulated Spike-Timing-Dependent Plasticity) is a biologically plausible learning rule for training Spiking Neural Networks (SNNs) without backpropagation. This document explains how to use R-STDP for training the policy network.

## Why R-STDP?

### Biological Plausibility

Unlike backpropagation, R-STDP uses **local learning rules** that only require information available at each synapse:

- ✅ **No global error signals** - no need to propagate errors backward
- ✅ **No symmetric weights** - forward and backward connections can differ
- ✅ **Local computation** - each synapse updates independently
- ✅ **Matches neuroscience** - similar to how real neurons learn

### Key Advantages

1. **Biologically Plausible**: Matches how biological neural networks learn
2. **Online Learning**: Learns during execution, not in separate training phase
3. **Hardware Compatible**: Works on neuromorphic hardware (no backprop needed)
4. **No PyTorch Required**: Pure NumPy implementation
5. **Real-time Adaptation**: Continuously adapts to changing conditions

## How R-STDP Works

### Three-Factor Learning Rule

R-STDP uses a three-factor learning rule:

```
Δw = learning_rate × reward × eligibility_trace
```

**Three Factors:**

1. **Pre-synaptic trace**: Tracks recent input spikes
   - Decays over time: `pre_trace *= pre_trace_decay`
   - Increases when input neuron spikes

2. **Post-synaptic trace**: Tracks recent output spikes
   - Decays over time: `post_trace *= post_trace_decay`
   - Increases when output neuron spikes

3. **Reward signal**: Task-dependent feedback
   - Positive = good behavior (strengthen synapses)
   - Negative = bad behavior (weaken synapses)
   - Zero = no learning (eligibility still decays)

**Eligibility Trace:**
```
eligibility = outer_product(post_trace, pre_trace)
```

This captures temporal correlation between pre and post spikes.

### Learning Process

1. **Forward Pass**: 
   - Input features → hidden spikes → output spikes
   - Update pre/post traces based on spikes
   - Update eligibility trace (pre × post)

2. **Reward Computation**:
   - Compute reward based on task performance
   - Examples: goal progress, obstacle avoidance, action smoothness

3. **Weight Update**:
   - Multiply eligibility by reward
   - Update weights: `w += learning_rate × reward × eligibility`
   - Clip weights to bounds

## Reward Function Design

The reward function is critical for R-STDP learning. It must provide clear feedback about task performance.

### Navigation Rewards

```python
from hippocampus_core.policy import NavigationRewardFunction, RewardConfig

config = RewardConfig(
    # Goal rewards
    goal_reward_gain=2.0,           # Reward per meter toward goal
    goal_reached_reward=10.0,        # Large reward when goal reached
    goal_reached_tolerance=0.1,      # Distance threshold for "reached"
    
    # Obstacle avoidance
    obstacle_penalty=-1.0,           # Penalty when too close
    obstacle_safety_margin=0.2,      # Safety margin (meters)
    collision_penalty=-5.0,          # Large penalty for collisions
    
    # Action smoothness
    angular_penalty_gain=0.2,       # Penalty for large angular velocities
    
    # General progress
    forward_progress_gain=0.5,      # Reward for any movement
    
    # Reward shaping
    reward_clip=10.0,               # Maximum absolute reward
    reward_scale=1.0,               # Overall scaling
)

reward_function = NavigationRewardFunction(config)
```

### Reward Components

1. **Goal Progress**: `reward += goal_reward_gain × (distance_reduction)`
   - Rewards reducing distance to goal
   - Encourages goal-seeking behavior

2. **Goal Reached**: `reward += goal_reached_reward` (if within tolerance)
   - Large positive reward when goal is reached
   - Signals task completion

3. **Obstacle Avoidance**: `reward += obstacle_penalty` (if too close)
   - Penalizes getting too close to obstacles
   - Encourages safe navigation

4. **Action Smoothness**: `reward -= angular_penalty_gain × |angular_velocity|`
   - Penalizes large angular velocities
   - Encourages smooth, stable motion

5. **Forward Progress**: `reward += forward_progress_gain × distance_moved`
   - Rewards any movement
   - Encourages exploration

## Training Workflow

### 1. Initialize Network

```python
from hippocampus_core.policy import RSTDPPolicySNN, RSTDPConfig

config = RSTDPConfig(
    feature_dim=44,        # Input feature dimension
    hidden_size=64,        # Number of hidden neurons
    output_size=2,         # Output actions (v, omega)
    learning_rate=5e-3,    # Learning rate
    hidden_decay=0.9,      # Membrane decay
    eligibility_decay=0.85,# Eligibility trace decay
)

network = RSTDPPolicySNN(config)
```

### 2. Create Policy Service

```python
from hippocampus_core.policy import (
    SpikingPolicyService,
    SpatialFeatureService,
    TopologyService,
    NavigationRewardFunction,
)

ts = TopologyService()
sfs = SpatialFeatureService(ts)
reward_fn = NavigationRewardFunction()

policy = SpikingPolicyService(
    feature_service=sfs,
    rstdp_model=network,
    reward_function=reward_fn,
)
```

### 3. Online Learning Loop

```python
for step in range(num_steps):
    # Get current state
    robot_state = get_robot_state()
    mission = get_mission()
    
    # Build features
    features, context = sfs.build_features(robot_state, mission)
    
    # Make decision (forward pass)
    decision = policy.decide(features, context, dt, mission)
    
    # Execute action
    execute_action(decision.action_proposal)
    
    # Compute reward
    reward = reward_fn.compute(robot_state, decision, mission, features, dt)
    
    # Update weights (happens automatically in policy.step())
    # Or manually: network.update_weights(reward)
    
    # Update robot state
    robot_state = update_state(decision, dt)
```

### 4. Monitor Learning

```python
# Track episode rewards
episode_reward = 0.0
for step in range(episode_steps):
    # ... learning loop ...
    episode_reward += reward

print(f"Episode reward: {episode_reward}")

# Check weight changes
weights = network.get_weights()
print(f"Weight statistics: min={weights['w_out'].min():.3f}, "
      f"max={weights['w_out'].max():.3f}, "
      f"mean={weights['w_out'].mean():.3f}")
```

## Hyperparameter Tuning

### Learning Rate

- **Too high**: Unstable learning, weights oscillate
- **Too low**: Slow learning, takes many episodes
- **Recommended**: Start with `5e-3`, adjust based on convergence

### Eligibility Decay

- **Higher (0.9-0.95)**: Longer memory, slower decay
- **Lower (0.7-0.8)**: Shorter memory, faster decay
- **Recommended**: `0.85` for navigation tasks

### Reward Scaling

- **Too large**: Unstable learning
- **Too small**: Slow learning
- **Recommended**: Clip rewards to `[-10, 10]` range

### Network Size

- **Hidden size**: 32-128 neurons (64 is good default)
- **Larger**: More capacity, slower learning
- **Smaller**: Faster learning, less capacity

## Comparison: R-STDP vs Backpropagation

| Aspect | R-STDP | Backpropagation |
|--------|--------|-----------------|
| **Biological Plausibility** | ✅ Yes | ❌ No |
| **Learning Type** | Online (during execution) | Offline (separate training) |
| **Information Required** | Local (at synapse) | Global (entire network) |
| **Hardware** | Neuromorphic compatible | Standard GPUs/CPUs |
| **Dependencies** | NumPy only | PyTorch, snnTorch |
| **Convergence** | Slower, but continuous | Faster, but requires training data |
| **Adaptation** | Real-time | Requires retraining |

## Best Practices

1. **Start with Random Weights**: R-STDP learns from scratch
2. **Design Good Rewards**: Clear, consistent reward signals
3. **Monitor Learning**: Track episode rewards and weight statistics
4. **Tune Hyperparameters**: Learning rate, decay rates, reward scaling
5. **Use Appropriate Network Size**: Balance capacity vs learning speed
6. **Reset Between Episodes**: Call `network.reset()` and `reward_fn.reset()`
7. **Save Checkpoints**: Use `network.get_weights()` to save learned weights

## Troubleshooting

### Learning Too Slow

- Increase learning rate (carefully)
- Increase reward magnitudes
- Check reward function (may be too sparse)

### Learning Unstable

- Decrease learning rate
- Clip rewards more aggressively
- Reduce reward magnitudes

### No Learning

- Check that rewards are non-zero
- Verify eligibility traces are updating (check for spikes)
- Ensure learning rate > 0

### Poor Performance

- Tune reward function (may need better reward shaping)
- Increase network size
- Adjust hyperparameters
- Check that rewards correlate with desired behavior

## Example: Complete Training Script

See `examples/rstdp_policy_demo.py` for a complete example of R-STDP training.

## References

- **STDP**: Spike-Timing-Dependent Plasticity (biological learning rule)
- **R-STDP**: Reward-modulated STDP (combines STDP with reward signal)
- **Three-Factor Learning**: Pre-spike, post-spike, and reward factors

## Summary

R-STDP provides a biologically plausible alternative to backpropagation for training SNNs. It learns online during execution using local learning rules, making it compatible with neuromorphic hardware and more aligned with biological neural networks.

