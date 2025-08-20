# SplitFed Learning Implementation

This branch implements **SplitFed Learning (SFL)** with 8 clients and 1 central server (9 computing nodes total). SplitFed Learning combines the privacy benefits of Split Learning with the efficiency advantages of Federated Learning, enabling parallel client training while maintaining data locality.

## 🎯 Algorithm Overview

### What is SplitFed Learning?
SplitFed Learning is an advanced distributed machine learning technique that enhances Split Learning by:
- **Parallel Training**: Multiple clients can train simultaneously (vs sequential in Split Learning)
- **Federated Aggregation**: Server aggregates model updates from multiple clients
- **Privacy Preservation**: Raw data remains local on client devices
- **Model Splitting**: Neural network split between clients and server at cut layer

### Key Innovations over Split Learning
1. **Parallel Execution**: Clients train concurrently rather than sequentially
2. **Model Aggregation**: Server averages client model updates using FedAvg
3. **Improved Efficiency**: Faster convergence through parallel processing
4. **Enhanced Scalability**: Better support for larger numbers of clients

### Architecture
```
Client 1:                Server:                 Client 2:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Local Data      │     │ Global Model    │     │ Local Data      │
│ ↓               │     │ Aggregation     │     │ ↓               │
│ ClientNN        │ →   │ & Updates       │ ←   │ ClientNN        │
│ (Cut Layer)     │     │ ServerNN        │     │ (Cut Layer)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ↓                        ↓                       ↓
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Local Training  │     │ FedAvg          │     │ Local Training  │
│ Model Updates   │ →   │ Aggregation     │ ←   │ Model Updates   │
└─────────────────┘     └─────────────────┘     └─────────────────┘

                    Parallel Training + Federated Aggregation
```

## 📁 Project Structure

### Core Implementation (`split_fed/`)
```
split_fed/
├── __init__.py          # Package initialization
├── client.py            # Enhanced client with federated capabilities
├── server.py            # Server with aggregation mechanisms
├── models/              # Model checkpoints storage
│   ├── global_client.pth    # Aggregated client model
│   ├── global_server.pth    # Aggregated server model
│   └── node_X_*.pth         # Individual node models
└── losses/              # Training progress tracking
    ├── training_losses/ # Per-round training losses
    └── scores.json      # Validation metrics
```

### Node Infrastructure (`nodes/`)
```
nodes/
├── global_var.py        # Neural network architectures (same as Split Learning)
├── model.py             # Model initialization and management
├── node0.py             # Server node (port 8000)
├── node1.py - node8.py  # Client nodes (ports 8001-8008)
└── data/                # FashionMNIST dataset storage
```

### Execution Scripts
```
run.py                   # Main execution script
stop.py                  # Graceful termination script
requirements.txt         # Python dependencies
```

## 🔬 Experimental Configuration

### Dataset: FashionMNIST (Same as Split Learning)
- **Images**: 28×28 grayscale fashion items
- **Classes**: 10 categories
- **Training**: 60,000 images distributed across 8 clients (~7,500 per client)
- **Testing**: 10,000 images distributed across 8 clients (~1,250 per client)
- **Distribution**: Deterministic IID distribution (ordered partitioning)

### Model Architecture (Identical to Split Learning)

#### Client-Side Model (ClientNN)
```python
class ClientNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28×28×1 → 28×28×32
        self.pool = nn.MaxPool2d(2, 2)                           # 28×28×32 → 14×14×32
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x = self.pool(self.relu(self.conv1(data)))               # Output: 32×14×14
        return x  # Cut layer activations
```

#### Server-Side Model (ServerNN)
```python
class ServerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14×14×32 → 14×14×64
        self.pool = nn.MaxPool2d(2, 2)                           # 14×14×64 → 7×7×64
        self.fc1 = nn.Linear(64 * 7 * 7, 128)                   # 3136 → 128
        self.fc2 = nn.Linear(128, 10)                            # 128 → 10 classes
        self.relu = nn.ReLU()
```

### Training Parameters
```yaml
# Training Configuration
rounds: 60                    # Total training rounds
local_epochs: 5               # Local epochs per client per round
batch_size: 128               # Mini-batch size
learning_rate: 3e-4           # Adam optimizer learning rate
device: "cpu"                 # Computation device

# Federated Configuration
aggregation_method: "FedAvg"  # Federated averaging
clients_per_round: 8          # All clients participate
parallel_training: true      # Multiple clients train simultaneously

# Network Configuration
server_port: 8000             # Server listening port
client_ports: 8001-8008       # Client listening ports
```

## 🚀 Running the Experiment

### Prerequisites
```bash
# Python 3.8+ required
python3 --version

# Install dependencies
pip install -r requirements.txt

# Verify FashionMNIST data
ls nodes/data/FashionMNIST/raw/
```

### Step-by-Step Execution

1. **Start the Experiment**
   ```bash
   python3 run.py
   ```
   
   This automatically:
   - Creates 9 node files with SplitFed logic
   - Launches server and 8 client processes
   - Initializes global models
   - Begins federated training

2. **Monitor Progress**
   ```bash
   # Server coordination logs
   tail -f logs/node_0.txt
   
   # Client training logs  
   tail -f logs/node_1.txt
   
   # Training losses
   watch -n 5 "ls -la split_fed/losses/"
   
   # Current validation scores
   tail split_fed/losses/scores.json
   ```

3. **Stop the Experiment**
   ```bash
   python3 stop.py
   ```

## 🔄 SplitFed Learning Process

### Training Flow (Key Differences from Split Learning)

#### Traditional Split Learning (Sequential)
```
Round 1: Client 1 → Client 2 → Client 3 → ... → Client 8
Round 2: Client 1 → Client 2 → Client 3 → ... → Client 8
```

#### SplitFed Learning (Parallel + Aggregation)
```
Round 1: Client 1, 2, 3, ..., 8 (ALL PARALLEL)
         ↓
         Server Aggregates Updates
         ↓
Round 2: Client 1, 2, 3, ..., 8 (ALL PARALLEL)
         ↓
         Server Aggregates Updates
```

### Detailed Training Protocol
```python
for round in range(total_rounds):
    # 1. PARALLEL CLIENT TRAINING
    for each client in parallel:
        client_model = load_global_client_model()
        for epoch in range(local_epochs):
            for batch in client_data:
                # Forward through client model
                activations = client_model(batch_data)
                
                # Send to server for completion
                server_response = server.train(activations, labels)
                
                # Update client model with gradients
                client_model.backward(server_response.gradients)
                client_optimizer.step()
        
        # Save updated client model
        save_client_model(client_model)
    
    # 2. FEDERATED AGGREGATION
    server.aggregate_client_models()  # FedAvg on client models
    server.aggregate_server_models()  # FedAvg on server models
    
    # 3. EVALUATION
    server.evaluate_global_model()
    
    # 4. BROADCAST UPDATED GLOBAL MODELS
    broadcast_global_models_to_clients()
```

### Federated Aggregation Implementation
```python
def aggregate(self):
    """FedAvg aggregation of server models"""
    clients = list(self.models.keys())
    weights_avg = copy.deepcopy(self.models[clients[0]].state_dict())
    
    # Average all client server models
    for k in weights_avg.keys():
        for i in range(1, len(clients)):
            weights_avg[k] += self.models[clients[i]].state_dict()[k]
        weights_avg[k] = torch.div(weights_avg[k], len(clients))
    
    self.avg_model.load_state_dict(weights_avg)

def aggregate_clients(self):
    """FedAvg aggregation of client models"""
    models = [torch.load(f"split_fed/models/node_{client_port-8000}_client.pth")
             for client_port in self.clients]
    
    global_client = {}
    for key in models[0].keys():
        global_client[key] = sum(model[key] for model in models) / len(models)
    
    torch.save(global_client, "./models/global_client.pth")
```

## 📊 Key Differences from Split Learning

### Performance Improvements
| Aspect | Split Learning | SplitFed Learning |
|--------|----------------|-------------------|
| **Training Mode** | Sequential | Parallel |
| **Round Time** | 8 × single_client_time | ~single_client_time |
| **Convergence** | Slower | Faster |
| **Scalability** | Poor (linear scaling) | Better (sub-linear scaling) |
| **Communication** | Per-batch per client | Per-round per client |

### Algorithmic Enhancements
1. **Parallel Processing**: All clients train simultaneously
2. **Model Aggregation**: Both client and server models aggregated
3. **Global Model Sharing**: Updated global models broadcast to all clients
4. **Synchronized Rounds**: Clear round-based training structure

### Implementation Differences
1. **Data Distribution**: Deterministic ordered partitioning vs random sampling
2. **Model Management**: Separate client and server model checkpoints
3. **Coordination**: Round-based synchronization vs sequential processing
4. **Aggregation**: Explicit FedAvg implementation

## 📈 Expected Results

### Performance Targets
- **Accuracy**: Similar to Split Learning (~85-90%) but faster convergence
- **Training Time**: ~50% faster than Split Learning due to parallelization
- **Communication Efficiency**: Reduced frequency (per-round vs per-batch)
- **Scalability**: Better support for larger client numbers

### Key Metrics to Monitor
```bash
# Training efficiency
grep "Round.*completed" logs/node_0.txt

# Convergence speed
python3 -c "
import json
with open('split_fed/losses/scores.json') as f:
    scores = json.load(f)
print(f'Convergence round (>80%): {next(i for i, acc in enumerate(scores) if acc > 0.8)}')
"

# Model checkpoints
ls -la split_fed/models/
```

## 🔧 Configuration and Customization

### Adjusting Aggregation Strategy
```python
# In split_fed/server.py - modify aggregate() method
# Example: Weighted averaging based on data size
def aggregate(self):
    clients = list(self.models.keys())
    data_sizes = self.get_client_data_sizes()
    total_size = sum(data_sizes.values())
    
    weights_avg = {}
    for k in self.models[clients[0]].state_dict().keys():
        weights_avg[k] = torch.zeros_like(self.models[clients[0]].state_dict()[k])
        for client in clients:
            weight = data_sizes[client] / total_size
            weights_avg[k] += weight * self.models[client].state_dict()[k]
```

### Modifying Client Selection
```python
# In split_fed/server.py - implement client sampling
def select_clients(self, fraction=0.5):
    """Select random subset of clients for each round"""
    num_selected = max(1, int(fraction * len(self.clients)))
    return random.sample(self.clients, num_selected)
```

### Data Distribution Strategies
```python
# In split_fed/client.py - modify get_data() for non-IID
def get_data_non_iid(self, alpha=0.5):
    """Create non-IID data distribution using Dirichlet"""
    # Implement Dirichlet distribution for class imbalance
    pass
```

## 🔍 Research Applications

### Baseline for Advanced Approaches
This SplitFed implementation serves as an enhanced baseline for:
- **SSFL (Sharded SplitFed)**: Adding hierarchical sharding
- **BSFL (Blockchain SplitFed)**: Adding blockchain security
- **Attack scenarios**: Testing robustness against adversarial clients

### Research Questions Addressed
1. **Efficiency**: How much faster is parallel vs sequential training?
2. **Scalability**: How does performance change with client count?
3. **Convergence**: Does federated aggregation improve convergence?
4. **Privacy**: What privacy-utility trade-offs exist?

### Comparative Metrics
- **Training Speed**: Wall-clock time comparison with Split Learning
- **Communication Cost**: Total bytes transmitted per round
- **Model Quality**: Final accuracy and convergence rate
- **Resource Usage**: CPU and memory consumption per client

## 🐛 Troubleshooting

### Common Issues
1. **Model Synchronization**: Ensure all clients complete before aggregation
2. **Memory Usage**: Parallel training increases memory requirements
3. **Port Conflicts**: Multiple simultaneous connections may conflict
4. **Model Checkpoint Corruption**: Verify checkpoint integrity

### Debug Tips
```bash
# Check all clients are training
for i in {1..8}; do
  echo "Client $i status:"
  curl -s http://localhost:800$i/status || echo "Not responding"
done

# Monitor model files
watch -n 2 "ls -la split_fed/models/"

# Check aggregation logs
grep -i "aggregate" logs/node_0.txt
```

## 📚 References

- **Split Learning**: Vepakomma et al. (2018)
- **Federated Learning**: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- **SplitFed Learning**: Thapa et al. "SplitFed: When Federated Learning Meets Split Learning" (2020)

---

**Note**: This SplitFed Learning implementation demonstrates the benefits of combining split learning's privacy preservation with federated learning's parallel efficiency, serving as a foundation for more advanced approaches like SSFL and BSFL.
