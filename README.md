# Sharded SplitFed Learning (SSFL)

This branch implements the Sharded SplitFed Learning (SSFL) algorithm with 36 nodes, consisting of 6 shard servers and 30 clients, where each shard server is responsible for 5 clients. The framework employs a federated server to aggregate the updates from shard servers and clients, ensuring scalability and efficient global model training. This README explains the algorithm, its implementation, and the code structure.

## Code Structure

### `sharding_split_fed` Folder

This folder contains the core implementation of SSFL, including:

- **`server` and `client` Classes**: Represent the shard server and client entities in the SSFL framework.
- **`losses` Folder**: Contains the training loss values reported by the servers during training and the `aggregator.json` file that tracks validation loss values.
- **Malicious Client Logic**: The framework allows for malicious nodes to perform data poisoning attacks, disrupting the global model's performance. This behavior is implemented in the `attack` method of the `Client` class.

### `nodes` Folder

In our setup, a **node** refers to any entity in the distributed learning framework, including shard servers, clients, and the federated server.

- **Flask-Based Node Communication**: Each node operates as a Flask server, enabling inter-node communication for distributed training.
- **36 Nodes Configuration**: Includes 30 clients and 6 shard servers, each implemented as an independent process.
- **`fed_server` class**: Implements the federated server, which performs global aggregation of shard server and client models using the FedAvg algorithm.

### `run.py` and `stop.py`

- **`run.py`**: Initializes the SSFL framework, including all shard servers, clients, and the global aggregator. It starts the training experiment.
- **`stop.py`**: Gracefully terminates all processes associated with the experiment.

## Federated Server (`fed_server.py`)
The `fed_server` acts as the global aggregator in the SSFL framework, performing the following key functions:

1. **Node Assignment**: Dynamically assigns shard servers and clients to specific nodes.
2. **FedAvg Aggregation**: Aggregates the models from shard servers and clients using the FedAvg algorithm to compute the global client and server models.
3. **Evaluation**: Evaluates the global models on the test dataset after each training cycle.
4. **Loss Logging**: Stores validation loss values in `aggregator.json` for performance tracking.

## Running the Experiment

To run the SSFL experiment, follow these steps:

1. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt

2. Start the experiment by running:
   ```bash
   python3 run.py
   ```
   This will spawn processes for all nodes (clients, including malicious nodes, and the central server) to execute the training process.

3. If needed, terminate the experiment manually using:
   ```bash
   python3 stop.py
   ```

## Testing and Evaluation
The experiment evaluates the scalability, performance, and security of the SSFL framework. The results are stored as follows:

- **Shard Server and Client Losses**: Stored in the `losses` folder.
- **Global Aggregated Losses**: Stored in the `aggregator.json` file within the `losses` folder.

This setup allows you to explore the benefits of SSFL, including reduced communication overhead, enhanced scalability, and resilience to data poisoning attacks.
