# Blockchain-Enabled SplitFed Learning (BSFL)
This branch implements the Blockchain-Enabled SplitFed Learning (BSFL) algorithm with 9 nodes, consisting of 3 shard servers and 6 clients, where each shard server is responsible for 2 clients. BSFL enhances distributed learning by replacing centralized servers with a blockchain-based consensus mechanism, ensuring secure, fair, and resilient global model training. This README explains the BSFL framework, its implementation, and the repository structure.


## Code Structure

### `blockchain_split_fed` Folder

This folder contains the core implementation of BSFL, including:

- **`client.py` and `server.py`**: Represent the Client and Server classes for BSFL nodes, facilitating model training, blockchain interactions, and communication within the framework.
- **`losses` Folder**: Contains the training loss values reported by the servers during training and the `aggregator.json` file that tracks validation loss values.

### Chaincode Folders

1. **`manage-scores/`**:
   - Implements the logic for tracking and managing the scores of nodes throughout the training process. 

2. **`model-propose/`**:
   - Includes methods to supervise and validate submitted models before integrating them into the blockchain.

3. **`eval-propose/`**:
   - Responsible for evaluating the performance of models and ensuring harmful contributions are excluded.

### `nodes` Folder

In the BSFL framework, a **node** refers to any entity in the system, including the global model, the aggregator, and Flask-based server-client nodes.

- **Node Implementation**: 
   - Includes processes for lightweight Flask-based communication.
   - Contains the aggregator responsible for combining updates from clients and shard servers into the global model.

### `express-application` Folder

- **`app1.js`**:
   - Implements a gateway using Express.js for communicating with the Hyperledger Fabric network.

### `test-network` Folder

- The Hyperledger Fabric test network, providing a blockchain environment to simulate decentralized training and validation workflows.

### `logs` Folder

- Contains logs capturing runtime operations, errors, training progress, and blockchain interactions.

### `run.py` and `stop.py`

1. **`run.py`**:
   - Orchestrates the BSFL framework by initializing nodes, shard servers, and the blockchain network. It starts the training experiment, spawning multiple processes for simultaneous operations.

2. **`stop.py`**:
   - Gracefully terminates all processes in the system if the training needs to be stopped manually.



## Federated and Blockchain Features

### Blockchain Integration

The BSFL framework employs a blockchain-based consensus mechanism to replace traditional central servers. Key features include:

1. **Decentralization**:
   - Eliminates single points of failure by implementing a committee-based blockchain for model aggregation and validation.

2. **Smart Contracts**:
   - Includes chaincodes for model proposal (`model-propose`), score management (`manage-scores`), and evaluation (`eval-propose`) to automate and secure operations.

3. **Resilience to Attacks**:
   - Prevents data poisoning and model tampering through evaluation and validation mechanisms built into the blockchain.

### Global Model Training

1. **Simultaneous Training**:
   - The framework spawns processes for all nodes (clients, shard servers, and blockchain-based global aggregators) to ensure efficient training.

2. **Fairness and Security**:
   - Model updates are evaluated to exclude malicious contributions, ensuring robust global model performance.



## Installation

### Prerequisites
- Python 3.x
- Node.js
- Hyperledger Fabric
- Docker

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/icdcs249/SSFL-BSFL
   cd SSFL-BSFL

2. Install dependencies for Python:
    ```bash
    pip install -r requirements.txt

3. Install dependencies for Nodejs in `eval-propose`, `express-application`, `manage-scores`, and `model-propose` folders:
    ```bash
    cd each_directory
    npm install

3. Set up the Hyperledger Fabric test network:
    To install Hyperledger Fabric, follow the commands mentioned [here](https://hyperledger-fabric.readthedocs.io/en/release-2.5/getting_started.html) to install the required docker images and binary files. Then, copy and paste the `bin` and `config` folders from the `fabric-samples` directory to this repo.

### Usage
1. Start the system:
    ```bash
    python3 run.py
    ```
    This will initiate the BSFL nodes, aggregators, and blockchain interactions for distributed learning.

2. Monitor logs in the `logs/` directory to track system operations and debugging.

3. To stop the system gracefully:
    ```bash
    python3 stop.py

## Testing and Evaluation
The experiment evaluates the scalability, performance, and security of the BSFL framework. The results are stored as follows:

- **Shard Server and Client Losses**: Stored in the losses folder.
- **Global Aggregated Losses**: Stored in the aggregator.json file within the losses folder.

This setup allows you to explore the benefits of BSFL, including reduced communication overhead, enhanced scalability, and resilience to data poisoning attacks.