# SplitFed Learning Methodology

## 🎯 Algorithm Overview

### Definition
SplitFed Learning (SFL) is a hybrid distributed machine learning approach that combines the **privacy-preserving properties of Split Learning** with the **parallel efficiency advantages of Federated Learning**. It enables multiple clients to train simultaneously while maintaining data locality and privacy.

### Core Innovation
The key innovation of SplitFed Learning over traditional Split Learning is the introduction of **parallel client training with federated aggregation**, replacing the sequential training approach with coordinated parallel execution.

```
Traditional Split Learning:        SplitFed Learning:
Client₁ → Server → Client₂ →      Client₁ ┐
Client₃ → Server → ... →          Client₂ ├─→ Server (Aggregation)
(Sequential)                      Client₃ ┘
                                 (Parallel + Federation)
```

## 🏗️ Architecture

### System Components

#### 1. Client Nodes (8 nodes)
- **Ports**: 8001-8008
- **Role**: Local model training on private data
- **Capabilities**:
  - Local data preprocessing
  - Client-side model forward pass
  - Gradient computation and model updates
  - Model checkpoint management

#### 2. Server Node (1 node)
- **Port**: 8000
- **Role**: Coordination and aggregation
- **Capabilities**:
  - Server-side model completion
  - Federated model aggregation (FedAvg)
  - Global model maintenance
  - Validation and evaluation

### Model Architecture Split

#### Client-Side Network (ClientNN)
```python
Input: 28×28×1 (FashionMNIST images)
│
├─── Conv2D(1→32, 3×3, padding=1)
├─── ReLU activation
├─── MaxPool2D(2×2)
│
Output: 14×14×32 (Cut layer activations)
```

**Mathematical Formulation:**
```
h₁ = MaxPool(ReLU(Conv₁(x)))
where x ∈ ℝ^(28×28×1), h₁ ∈ ℝ^(14×14×32)
```

#### Server-Side Network (ServerNN)
```python
Input: 14×14×32 (from client cut layer)
│
├─── Conv2D(32→64, 3×3, padding=1)
├─── ReLU activation
├─── MaxPool2D(2×2)
├─── Flatten: 64×7×7 → 3136
├─── Linear(3136→128)
├─── ReLU activation
├─── Linear(128→10)
│
Output: 10 (class logits)
```

**Mathematical Formulation:**
```
h₂ = MaxPool(ReLU(Conv₂(h₁)))
f = flatten(h₂)
z = Linear₂(ReLU(Linear₁(f)))
where z ∈ ℝ^10 (class scores)
```

## 🔄 Training Protocol

### Round-Based Training Flow

#### Phase 1: Parallel Client Training
```python
# For each round r ∈ {1, 2, ..., R}
for round_r in range(total_rounds):
    # All clients train in parallel
    parallel_for client_i in clients:
        # Download global models
        θ_client^(r) = download_global_client_model()
        θ_server_i^(r) = download_global_server_model()
        
        # Local training for E epochs
        for epoch in range(local_epochs):
            for batch in local_dataset_i:
                # Forward pass
                h_i = client_model(x_batch; θ_client^(r))
                
                # Send activations to server
                logits = server.forward(h_i; θ_server_i^(r))
                loss = CrossEntropy(logits, y_batch)
                
                # Backward pass
                ∇h_i = server.backward(loss)
                ∇θ_client = client.backward(∇h_i)
                
                # Update local models
                θ_client^(r) ← θ_client^(r) - η∇θ_client
                θ_server_i^(r) ← θ_server_i^(r) - η∇θ_server
        
        # Save updated models
        save_local_models(θ_client^(r), θ_server_i^(r))
```

#### Phase 2: Federated Aggregation
```python
# Server aggregates all client models using FedAvg
def federated_averaging():
    # Aggregate client models
    θ_client^(r+1) = (1/N) * Σᵢ₌₁ᴺ θ_client_i^(r)
    
    # Aggregate server models  
    θ_server^(r+1) = (1/N) * Σᵢ₌₁ᴺ θ_server_i^(r)
    
    # Broadcast updated global models
    broadcast_global_models(θ_client^(r+1), θ_server^(r+1))
```

### Mathematical Framework

#### Objective Function
The global objective in SplitFed Learning is:
```
min F(θ_client, θ_server) = (1/N) Σᵢ₌₁ᴺ Fᵢ(θ_client, θ_server)

where Fᵢ(θ_client, θ_server) = (1/|Dᵢ|) Σ_(x,y)∈Dᵢ ℓ(f(x; θ_client, θ_server), y)
```

#### Local Update Rule
Each client performs local SGD updates:
```
θ_client^(t+1) = θ_client^(t) - η∇_θ_client Fᵢ(θ_client^(t), θ_server_i^(t))
θ_server_i^(t+1) = θ_server_i^(t) - η∇_θ_server Fᵢ(θ_client^(t), θ_server_i^(t))
```

#### Global Aggregation
FedAvg aggregation with equal weights:
```
θ_client^(r+1) = (1/N) Σᵢ₌₁ᴺ θ_client_i^(r)
θ_server^(r+1) = (1/N) Σᵢ₌₁ᴺ θ_server_i^(r)
```

## 🔄 Communication Protocol

### Client-Server Interaction

#### Training Request Flow
```
1. Client sends activations: POST /train
   Data: {
     "activations": h_i,      # Cut layer outputs
     "labels": y_batch,       # Ground truth labels
     "round": r,              # Current round number
     "client_id": i           # Client identifier
   }

2. Server processes and responds:
   Response: {
     "gradients": ∇h_i,       # Gradients for client backprop
     "loss": loss_value,      # Batch loss for monitoring
     "status": "success"      # Processing status
   }
```

#### Model Synchronization
```
1. Download Global Models: GET /models
   Response: {
     "client_model": θ_client^(r),     # Global client weights
     "server_model": θ_server^(r),     # Global server weights
     "round": r                        # Current round
   }

2. Upload Local Models: POST /upload_model
   Data: {
     "client_model": θ_client_i^(r),   # Updated client weights
     "server_model": θ_server_i^(r),   # Updated server weights
     "client_id": i,                   # Client identifier
     "round": r                        # Training round
   }
```

### Aggregation Protocol
```python
def aggregation_protocol():
    # 1. Wait for all clients to complete training
    wait_for_all_clients()
    
    # 2. Collect all model updates
    client_models = collect_client_models()
    server_models = collect_server_models()
    
    # 3. Perform FedAvg aggregation
    global_client_model = fedavg(client_models)
    global_server_model = fedavg(server_models)
    
    # 4. Update global models
    save_global_models(global_client_model, global_server_model)
    
    # 5. Evaluate global model
    accuracy = evaluate_global_model()
    
    # 6. Log progress
    log_round_metrics(round, accuracy, loss)
```

## 📊 Data Distribution Strategy

### IID Distribution (Current Implementation)
```python
def distribute_data_iid(dataset, num_clients):
    """Deterministic ordered partitioning for reproducibility"""
    data_per_client = len(dataset) // num_clients
    client_datasets = {}
    
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client
        client_datasets[i] = dataset[start_idx:end_idx]
    
    return client_datasets

# Example distribution for FashionMNIST:
# Client 1: samples 0-7499      (T-shirt, Trouser, ...)
# Client 2: samples 7500-14999  (...)
# Client 8: samples 52500-59999 (...)
```

### Statistical Properties
```
Training Data Distribution:
- Total samples: 60,000
- Per client: 7,500 samples
- Classes per client: All 10 classes (balanced)
- Distribution type: IID (ordered partitioning)

Testing Data Distribution:
- Total samples: 10,000  
- Per client: 1,250 samples
- Evaluation: Aggregated across all clients
```

## 🚀 Performance Analysis

### Theoretical Complexity

#### Communication Complexity
```
Per Round Communication:
- Client → Server: |h_i| × batch_size × batches_per_epoch × local_epochs
- Server → Client: |h_i| × batch_size × batches_per_epoch × local_epochs
- Model Exchange: |θ_client| + |θ_server| (once per round)

Total per round: O(|h_i| × local_data_size + |θ|)
```

#### Computational Complexity
```
Client Computation per Round:
- Forward: O(|θ_client| × local_data_size)
- Backward: O(|θ_client| × local_data_size)

Server Computation per Round:
- Forward/Backward: O(|θ_server| × total_data_size)
- Aggregation: O(|θ| × num_clients)
```

### Parallelization Benefits
```
Traditional Split Learning:
- Round time: Σᵢ₌₁ᴺ training_time_i
- Total time: rounds × Σᵢ₌₁ᴺ training_time_i

SplitFed Learning:
- Round time: max(training_time_i) + aggregation_time
- Total time: rounds × (max(training_time_i) + aggregation_time)

Speedup Factor: N / (1 + aggregation_overhead)
Expected speedup: ~4-6x for 8 clients
```

## 🔒 Privacy Analysis

### Privacy Guarantees

#### Data Privacy
1. **Local Data Remains Private**: Raw data never leaves client devices
2. **Intermediate Representations**: Only cut layer activations shared
3. **No Label Leakage**: Labels processed locally on server during training

#### Model Privacy
1. **Partial Model Sharing**: Clients only receive global client model
2. **Server Model Protection**: Server portion remains centralized
3. **Aggregation Privacy**: Individual updates aggregated via FedAvg

### Privacy vs Utility Trade-offs
```
Privacy Level: High (split model + local data)
Utility Level: High (full model expressiveness)
Communication: Medium (activations + gradients per batch)

Comparison with alternatives:
- vs Federated Learning: Higher privacy (no full model sharing)
- vs Split Learning: Similar privacy, better efficiency
- vs Centralized: Much higher privacy, comparable utility
```

## 🎯 Convergence Analysis

### Theoretical Convergence
Under standard assumptions (convex loss, bounded gradients), SplitFed Learning converges to:
```
E[F(θ^T)] - F* ≤ O(1/T) + O(E²η²γ²)

where:
- T: total iterations
- E: local epochs per round
- η: learning rate  
- γ: data heterogeneity measure
```

### Practical Convergence Factors
1. **Local Epochs (E=5)**: Balance between local convergence and global consistency
2. **Learning Rate (η=3e-4)**: Conservative for stable convergence
3. **Model Architecture**: Appropriate capacity for FashionMNIST
4. **Data Distribution**: IID reduces convergence challenges

### Expected Performance
```yaml
Target Metrics:
  final_accuracy: ≥85%          # Similar to centralized training
  convergence_round: ≤40        # Faster than sequential Split Learning
  training_speedup: 4-6x        # vs Sequential Split Learning
  communication_efficiency: High # Per-round vs per-batch
```

## 🔧 Hyperparameter Sensitivity

### Critical Parameters

#### Learning Rate (η = 3e-4)
```python
# Sensitivity analysis
learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
# Expected: 3e-4 provides best stability vs convergence speed
```

#### Local Epochs (E = 5)
```python
# Trade-off analysis
local_epochs = [1, 3, 5, 10]
# Expected: E=5 balances local/global optimization
```

#### Batch Size (B = 128)
```python
# Memory vs gradient noise trade-off
batch_sizes = [32, 64, 128, 256]
# Expected: 128 provides good gradient estimates
```

### Model Architecture Sensitivity
```python
# Client model depth
client_layers = [1, 2, 3]  # Current: 1 conv layer
# Trade-off: More layers → more privacy, less efficiency

# Cut layer position
cut_positions = ["after_conv1", "after_conv2", "after_fc1"]
# Current: after_conv1 (early cut for efficiency)
```

## 📈 Evaluation Metrics

### Primary Metrics
1. **Accuracy**: Classification accuracy on test set
2. **Loss**: Cross-entropy loss convergence
3. **Training Time**: Wall-clock time per round
4. **Communication Cost**: Bytes transmitted per round

### Secondary Metrics
1. **Model Size**: Parameters in client vs server models
2. **Memory Usage**: Peak memory consumption per node
3. **Scalability**: Performance vs number of clients
4. **Robustness**: Stability under varying conditions

### Benchmark Comparisons
```yaml
Baselines:
  - Centralized Learning: Upper bound on accuracy
  - Federated Learning: Communication and privacy comparison
  - Split Learning: Direct efficiency comparison
  - Local Training: Lower bound performance

Evaluation Protocol:
  - Multiple random seeds (3-5 runs)
  - Statistical significance testing
  - Confidence intervals on metrics
  - Ablation studies on key components
```

## 🔬 Research Extensions

### Immediate Extensions
1. **Non-IID Data**: Implement Dirichlet distribution for heterogeneity
2. **Client Sampling**: Random client selection each round
3. **Adaptive Aggregation**: Weighted averaging based on data size
4. **Differential Privacy**: Add noise to gradients/models

### Advanced Extensions
1. **Personalization**: Client-specific model adaptations
2. **Compression**: Gradient/model compression techniques
3. **Asynchronous Training**: Remove synchronization requirements
4. **Hierarchical Aggregation**: Multi-level federated structure

### Research Questions
1. **Optimal Cut Layer**: Where should the model be split?
2. **Privacy-Utility Trade-off**: Quantitative privacy analysis
3. **Heterogeneity Robustness**: Performance under non-IID data
4. **Scalability Limits**: Maximum number of clients supported

---

**Note**: This methodology forms the foundation for understanding SplitFed Learning and serves as a baseline for advanced variants like SSFL (Sharded SplitFed) and BSFL (Blockchain SplitFed) implemented in other branches.
