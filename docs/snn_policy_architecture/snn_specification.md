# SNN Policy Specification: snnTorch Integration

This document specifies the Spiking Neural Network (SNN) implementation details for the policy system, aligned with snnTorch best practices and the existing codebase patterns.

## 1. Overview

The SNN Policy Service (SPS) uses **snnTorch** (https://snntorch.readthedocs.io/) as the spiking neural network framework. snnTorch is built on PyTorch and provides:

- Leaky Integrate-and-Fire (LIF) neurons
- Surrogate gradient descent for training
- Efficient GPU/CPU inference
- Integration with PyTorch's ecosystem

**References**:
- snnTorch documentation: https://snntorch.readthedocs.io/
- snnTorch tutorials: https://github.com/snntorch/Spiking-Neural-Networks-Tutorials
- Existing implementation: `src/hippocampus_core/controllers/snntorch_controller.py`

## 2. Spike Encoding

### 2.1 Encoding Schemes

Features must be converted from continuous values to spike trains. snnTorch provides encoding via `snntorch.spikegen`.

**Location**: `src/hippocampus_core/policy/spike_encoding.py`

#### 2.1.1 Rate Coding (Default)

**Use case**: Most features (goal-relative, neighbors, safety bands)

```python
import snntorch.spikegen as spikegen
import torch

def encode_rate(
    features: torch.Tensor,
    num_steps: int = 1,
    gain: float = 1.0,
    offset: float = 0.0,
) -> torch.Tensor:
    """Encode features using rate coding.
    
    Parameters
    ----------
    features:
        Tensor of shape (batch, feature_dim) with values in [0, 1] or normalized.
    num_steps:
        Number of time steps to encode (default: 1 for single-step inference).
    gain:
        Gain factor to scale firing rates.
    offset:
        Offset to shift firing rates.
        
    Returns
    -------
    torch.Tensor
        Spike train of shape (num_steps, batch, feature_dim).
    """
    # Normalize features to [0, 1] if needed
    normalized = torch.clamp(features * gain + offset, 0.0, 1.0)
    
    # Generate Poisson spike trains
    spike_train = spikegen.rate(
        normalized.unsqueeze(0).repeat(num_steps, 1, 1),
        num_steps=num_steps,
    )
    return spike_train
```

**Usage in policy**:
- **Single-step inference** (real-time control): `num_steps=1`
- **Multi-step inference** (temporal integration): `num_steps=5-10` for better temporal context

#### 2.1.2 Latency Coding

**Use case**: Time-sensitive features (obstacle proximity, urgency)

```python
def encode_latency(
    features: torch.Tensor,
    num_steps: int = 10,
    threshold: float = 0.01,
    gain: float = 1.0,
) -> torch.Tensor:
    """Encode features using latency coding.
    
    Higher values spike earlier in the sequence.
    """
    normalized = torch.clamp(features * gain, threshold, 1.0)
    spike_train = spikegen.latency(
        normalized.unsqueeze(0).repeat(num_steps, 1, 1),
        num_steps=num_steps,
        threshold=threshold,
    )
    return spike_train
```

#### 2.1.3 Delta Modulation

**Use case**: Change detection (velocity, acceleration)

```python
def encode_delta(
    features: torch.Tensor,
    features_prev: torch.Tensor,
    num_steps: int = 1,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Encode feature changes using delta modulation."""
    delta = features - features_prev
    spike_train = spikegen.delta(
        delta.unsqueeze(0).repeat(num_steps, 1, 1),
        threshold=threshold,
    )
    return spike_train
```

### 2.2 Feature-to-Spike Pipeline

**Location**: `src/hippocampus_core/policy/spike_encoding.py`

```python
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
from .features import FeatureVector

@dataclass
class EncodingConfig:
    """Configuration for spike encoding."""
    encoding_scheme: str = "rate"  # "rate", "latency", "delta"
    num_steps: int = 1  # Time steps for encoding
    gain: float = 1.0
    offset: float = 0.0
    threshold: float = 0.01  # For latency/delta encoding

class FeatureEncoder:
    """Encodes FeatureVector to spike trains."""
    
    def __init__(self, config: EncodingConfig):
        self.config = config
        self._prev_features: Optional[torch.Tensor] = None
        
    def encode(
        self,
        features: FeatureVector,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Convert FeatureVector to spike train.
        
        Returns
        -------
        torch.Tensor
            Spike train of shape (num_steps, batch=1, feature_dim).
        """
        # Convert FeatureVector to tensor
        feature_tensor = self._vector_to_tensor(features, device)
        
        # Apply encoding scheme
        if self.config.encoding_scheme == "rate":
            spike_train = encode_rate(
                feature_tensor,
                num_steps=self.config.num_steps,
                gain=self.config.gain,
                offset=self.config.offset,
            )
        elif self.config.encoding_scheme == "latency":
            spike_train = encode_latency(
                feature_tensor,
                num_steps=self.config.num_steps,
                threshold=self.config.threshold,
                gain=self.config.gain,
            )
        elif self.config.encoding_scheme == "delta":
            if self._prev_features is None:
                # Fallback to rate on first step
                spike_train = encode_rate(feature_tensor, num_steps=self.config.num_steps)
            else:
                spike_train = encode_delta(
                    feature_tensor,
                    self._prev_features,
                    num_steps=self.config.num_steps,
                    threshold=self.config.threshold,
                )
            self._prev_features = feature_tensor.clone()
        else:
            raise ValueError(f"Unknown encoding scheme: {self.config.encoding_scheme}")
        
        return spike_train
    
    def _vector_to_tensor(self, features: FeatureVector, device: torch.device) -> torch.Tensor:
        """Convert FeatureVector to flat tensor."""
        # Concatenate all feature components
        feature_list = []
        feature_list.extend(features.goal_ego)
        for neighbor in features.neighbors_k:
            feature_list.extend(neighbor)
        feature_list.extend(features.topo_ctx)
        feature_list.extend(features.safety)
        if features.dynamics:
            feature_list.extend(features.dynamics)
        
        return torch.tensor(feature_list, dtype=torch.float32, device=device).unsqueeze(0)  # (1, dim)
    
    def reset(self) -> None:
        """Reset encoder state (e.g., for new episode)."""
        self._prev_features = None
```

## 3. Neuron Model: Leaky Integrate-and-Fire (LIF)

### 3.1 LIF Neuron Dynamics

snnTorch provides `snn.Leaky` for LIF neurons. The membrane potential evolves as:

```
V(t+1) = β * V(t) + I(t)
```

Where:
- `β` is the decay factor (0 < β < 1)
- `I(t)` is the input current
- When `V(t) ≥ threshold`, a spike is emitted and `V(t) = reset`

### 3.2 Network Architecture

**Location**: `src/hippocampus_core/policy/snn_network.py`

Following the pattern from `SnnControllerNet`:

```python
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Optional
from ..controllers.snntorch_controller import resolve_surrogate

class PolicySNN(nn.Module):
    """SNN network for policy decisions.
    
    Architecture:
    - Input layer: Linear(feature_dim, hidden_dim)
    - Hidden layer: LIF neurons
    - Output layer: Linear(hidden_dim, output_dim)
    - Readout: tanh activation for continuous actions
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,  # [v, ω] or [v, ω, vz] for 3D
        beta: float = 0.9,
        spike_grad: str = "atan",
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.fc_in = nn.Linear(feature_dim, hidden_dim)
        
        # LIF neuron layer
        surrogate_fn = resolve_surrogate(spike_grad)
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=surrogate_fn,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        )
        
        # Output readout
        self.readout = nn.Linear(hidden_dim, output_dim)
        
    def forward_step(
        self,
        spike_input: torch.Tensor,  # (batch, feature_dim) - already encoded
        membrane: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward pass.
        
        Parameters
        ----------
        spike_input:
            Input spikes (batch, feature_dim).
        membrane:
            Previous membrane potential (batch, hidden_dim).
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (action, next_membrane) where action is (batch, output_dim).
        """
        batch_size = spike_input.size(0)
        if membrane is None:
            membrane = torch.zeros(batch_size, self.hidden_dim, device=spike_input.device)
        
        # Project input spikes to currents
        currents = self.fc_in(spike_input)
        
        # LIF neuron dynamics
        spikes, membrane = self.lif(currents, membrane)
        
        # Readout from membrane potential (not spikes)
        # This provides smooth continuous output
        action_logits = self.readout(membrane)
        actions = torch.tanh(action_logits)  # Bound to [-1, 1]
        
        return actions, membrane
    
    def forward_sequence(
        self,
        spike_train: torch.Tensor,  # (time_steps, batch, feature_dim)
        membrane: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-step forward pass for temporal integration.
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (actions, final_membrane) where actions is (time_steps, batch, output_dim).
        """
        time_steps = spike_train.size(0)
        outputs = []
        next_membrane = membrane
        
        for t in range(time_steps):
            action, next_membrane = self.forward_step(spike_train[t], next_membrane)
            outputs.append(action)
        
        return torch.stack(outputs, dim=0), next_membrane
    
    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize membrane potential to zero."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)
```

### 3.3 Hyperparameters

**Default values** (aligned with existing `SnnControllerNet`):

```python
@dataclass
class SNNConfig:
    """Configuration for SNN network."""
    feature_dim: int
    hidden_dim: int = 64  # Increased from 32 for richer policy
    output_dim: int = 2  # [v, ω] for 2D, [v, ω, vz] for 3D
    beta: float = 0.9  # Membrane decay (0.9 = ~10ms time constant at 100Hz)
    spike_grad: str = "atan"  # Surrogate gradient function
    threshold: float = 1.0  # Spike threshold
    reset_mechanism: str = "subtract"  # "subtract" or "zero"
    num_steps: int = 1  # Time steps for encoding/inference
```

**Beta parameter**:
- `β = 0.9` at 100Hz → time constant ≈ 10ms
- `β = 0.95` → time constant ≈ 20ms (slower dynamics)
- `β = 0.8` → time constant ≈ 5ms (faster dynamics)

## 4. Decision Decoding

### 4.1 Spike-to-Action Decoding

The network outputs continuous actions from membrane potentials. For policy decisions:

**Location**: `src/hippocampus_core/policy/decision_decoding.py`

```python
from dataclasses import dataclass
import torch
import numpy as np
from .features import FeatureVector, LocalContext
from .mission import Mission

@dataclass
class ActionProposal:
    """Proposed action from SNN."""
    v: float  # Linear velocity (m/s)
    omega: float  # Angular velocity (rad/s)
    vz: Optional[float] = None  # Vertical velocity (m/s) for 3D

@dataclass
class PolicyDecision:
    """Policy decision output."""
    next_waypoint: Optional[int] = None  # Graph node ID or None for reactive
    action_proposal: ActionProposal
    confidence: float  # [0, 1]
    reason: str  # "snn", "heuristic", "fallback"

class DecisionDecoder:
    """Decodes SNN output to PolicyDecision."""
    
    def __init__(
        self,
        max_linear: float = 0.3,
        max_angular: float = 1.0,
        max_vertical: Optional[float] = None,
    ):
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.max_vertical = max_vertical
        
    def decode(
        self,
        snn_output: torch.Tensor,  # (batch, output_dim) or (time_steps, batch, output_dim)
        features: FeatureVector,
        local_context: LocalContext,
        mission: Mission,
    ) -> PolicyDecision:
        """Decode SNN output to policy decision.
        
        Parameters
        ----------
        snn_output:
            Network output, shape (batch, output_dim) or (time_steps, batch, output_dim).
            If multi-step, use the last timestep.
        features:
            Input features (for context).
        local_context:
            Local context (for waypoint selection).
        mission:
            Mission goals (for validation).
            
        Returns
        -------
        PolicyDecision
        """
        # Handle multi-step output
        if snn_output.dim() == 3:
            snn_output = snn_output[-1]  # Use last timestep
        
        # Extract actions (assumed to be in [-1, 1] from tanh)
        action_np = snn_output.squeeze(0).cpu().numpy()
        
        # Scale to physical units
        v = float(action_np[0] * self.max_linear)
        omega = float(action_np[1] * self.max_angular)
        vz = None
        if len(action_np) > 2 and self.max_vertical is not None:
            vz = float(action_np[2] * self.max_vertical)
        
        # Compute confidence from output magnitude
        # Higher magnitude = more confident
        output_magnitude = np.linalg.norm(action_np)
        confidence = float(np.clip(output_magnitude, 0.0, 1.0))
        
        # Select waypoint (if using hierarchical planning)
        waypoint = self._select_waypoint(features, local_context, mission)
        
        return PolicyDecision(
            next_waypoint=waypoint,
            action_proposal=ActionProposal(v=v, omega=omega, vz=vz),
            confidence=confidence,
            reason="snn",
        )
    
    def _select_waypoint(
        self,
        features: FeatureVector,
        local_context: LocalContext,
        mission: Mission,
    ) -> Optional[int]:
        """Select target waypoint from graph (optional, for hierarchical planning)."""
        # For reactive control (Milestone A), return None
        # For hierarchical planning (Milestone C+), implement waypoint selection
        return None
```

### 4.2 Confidence Estimation

Confidence can be estimated from:
1. **Output magnitude**: Larger outputs = higher confidence
2. **Spike rate**: Higher spike rates = more active decision
3. **Membrane stability**: Stable membrane = confident decision
4. **Feature quality**: High-quality features = higher confidence

```python
def compute_confidence(
    snn_output: torch.Tensor,
    spike_rate: Optional[float] = None,
    feature_quality: float = 1.0,
) -> float:
    """Compute confidence score from SNN outputs."""
    # Base confidence from output magnitude
    output_mag = torch.norm(snn_output).item()
    base_confidence = min(output_mag, 1.0)
    
    # Adjust by spike rate (if available)
    if spike_rate is not None:
        # Normalize spike rate to [0, 1]
        rate_confidence = min(spike_rate / 50.0, 1.0)  # Assume 50 Hz is "high"
        base_confidence = 0.7 * base_confidence + 0.3 * rate_confidence
    
    # Adjust by feature quality
    final_confidence = base_confidence * feature_quality
    
    return float(np.clip(final_confidence, 0.0, 1.0))
```

## 5. Temporal Context

### 5.1 History Buffers

For temporal integration, maintain history of:
- Feature vectors
- Decisions
- Membrane states

```python
from collections import deque
from typing import Deque

class TemporalContext:
    """Maintains temporal context for policy decisions."""
    
    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        self.feature_history: Deque[FeatureVector] = deque(maxlen=history_length)
        self.decision_history: Deque[PolicyDecision] = deque(maxlen=history_length)
        self.membrane_history: Deque[torch.Tensor] = deque(maxlen=history_length)
        
    def update(
        self,
        features: FeatureVector,
        decision: PolicyDecision,
        membrane: torch.Tensor,
    ) -> None:
        """Update history buffers."""
        self.feature_history.append(features)
        self.decision_history.append(decision)
        self.membrane_history.append(membrane.clone())
    
    def get_temporal_features(self) -> torch.Tensor:
        """Get concatenated temporal features for multi-step encoding."""
        if not self.feature_history:
            return None
        
        # Concatenate recent features
        feature_tensors = [
            self._vector_to_tensor(fv) for fv in self.feature_history
        ]
        return torch.stack(feature_tensors, dim=0)  # (time_steps, feature_dim)
    
    def reset(self) -> None:
        """Reset history."""
        self.feature_history.clear()
        self.decision_history.clear()
        self.membrane_history.clear()
```

### 5.2 Recurrent Connections (Future)

For richer temporal dynamics, add recurrent connections:

```python
class RecurrentPolicySNN(nn.Module):
    """SNN with recurrent connections for temporal memory."""
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, beta: float):
        super().__init__()
        self.fc_in = nn.Linear(feature_dim, hidden_dim)
        self.fc_recurrent = nn.Linear(hidden_dim, hidden_dim)  # Recurrent connection
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.atan())
        self.readout = nn.Linear(hidden_dim, output_dim)
    
    def forward_step(self, spike_input, membrane, recurrent_input):
        """Forward with recurrent input."""
        currents = self.fc_in(spike_input) + self.fc_recurrent(recurrent_input)
        spikes, membrane = self.lif(currents, membrane)
        actions = torch.tanh(self.readout(membrane))
        return actions, membrane, spikes  # Return spikes for recurrence
```

## 6. Training Interface (Future)

### 6.1 Data Format

For imitation learning or RL:

```python
@dataclass
class TrainingSample:
    """Single training sample."""
    features: FeatureVector
    action: ActionProposal  # Expert/ground truth action
    reward: Optional[float] = None  # For RL
    next_features: Optional[FeatureVector] = None  # For RL
    done: bool = False  # Episode termination
```

### 6.2 Loss Functions

**Imitation Learning** (MSE on actions):
```python
def imitation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss for imitation learning."""
    return nn.functional.mse_loss(predicted, target)
```

**Reinforcement Learning** (Policy gradient):
```python
def policy_gradient_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Policy gradient loss (PPO, A2C, etc.)."""
    return -(log_probs * advantages).mean()
```

### 6.3 Checkpoint Format

Follow existing checkpoint format from `SnnTorchController`:

```python
checkpoint = {
    "version": "1.0",
    "model_state": model.state_dict(),
    "model_hparams": {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "beta": beta,
        "spike_grad": spike_grad,
    },
    "obs_mean": obs_mean,  # Feature normalization
    "obs_std": obs_std,
    "action_scale": action_scale,  # [max_linear, max_angular]
    "time_steps": num_steps,
    "metadata": {
        "training_loss": final_loss,
        "validation_loss": val_loss,
        "epochs": num_epochs,
    },
}
```

## 7. Implementation Phases

### Milestone A: Heuristic Stub
- No SNN inference
- Direct feature → action mapping
- Validate pipeline

### Milestone B: SNN Runtime
- Implement `FeatureEncoder` with rate coding
- Implement `PolicySNN` network
- Implement `DecisionDecoder`
- Use pre-trained weights (if available) or random initialization
- Single-step inference (`num_steps=1`)

### Milestone C: Temporal Integration
- Multi-step encoding (`num_steps=5-10`)
- Temporal context buffers
- Optional: Recurrent connections

### Milestone D: Training
- Implement training loop
- Imitation learning from expert
- Optional: Reinforcement learning

## 8. Performance Considerations

### 8.1 Inference Speed

**Target**: <5ms p50, <10ms p99 at 10Hz

**Optimizations**:
- Use `torch.no_grad()` for inference
- Batch processing when possible
- TorchScript compilation for deployment
- GPU acceleration (if available)

### 8.2 Memory

- Detach membrane states between steps: `membrane.detach()`
- Clear history buffers periodically
- Use `torch.jit.script()` for reduced memory footprint

## 9. Integration with Existing Code

The SNN policy follows the same patterns as `SnnTorchController`:

```python
# Similar structure to SnnTorchController
class SpikingPolicyService(SNNController):
    def __init__(self, feature_service, snn_model, config):
        self._feature_service = feature_service
        self._encoder = FeatureEncoder(config.encoding)
        self._snn = snn_model  # PolicySNN instance
        self._decoder = DecisionDecoder(config.action_limits)
        self._membrane = None
        
    def reset(self):
        self._membrane = self._snn.init_state(1, self.device)
        self._encoder.reset()
        
    def step(self, obs, dt):
        # Build features
        features, context = self._feature_service.build_features(...)
        
        # Encode to spikes
        spikes = self._encoder.encode(features)
        
        # SNN inference
        action_tensor, self._membrane = self._snn.forward_step(
            spikes.squeeze(0),  # Remove time dimension for single-step
            self._membrane,
        )
        
        # Decode to decision
        decision = self._decoder.decode(action_tensor, features, context, mission)
        
        # Return action array
        return np.array([decision.action_proposal.v, decision.action_proposal.omega])
```

## 10. Summary

**Key Specifications**:

1. **Encoding**: Rate coding (default), latency/delta (optional)
2. **Neuron Model**: LIF with `β=0.9`, `threshold=1.0`
3. **Architecture**: Feedforward (hidden_dim=64) → LIF → Linear readout → tanh
4. **Decoding**: Membrane potential → continuous actions → scaled to physical units
5. **Temporal**: History buffers (Milestone B+), recurrent connections (optional)
6. **Training**: Imitation learning (MSE), RL (policy gradient) - future
7. **Performance**: <5ms inference, TorchScript for deployment

This specification aligns with snnTorch best practices and integrates seamlessly with the existing `SnnTorchController` patterns.

