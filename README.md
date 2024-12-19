# SplitFed Learning (Attacked Scenario with Malicious Nodes)

This branch implements the SplitFed Learning (SFL) algorithm with 8 clients, 1 central server, totaling 9 computing nodes. Among the clients, 3 are malicious and perform data poisoning attacks to degrade the global model. The following explains the algorithm, its implementation, and how the attacked scenario is set up.

## Code Structure

### `split_fed` Folder

This folder contains the core implementation of SplitFed Learning, including:  
- **`server` and `client` classes**: Represent the server and client entities in the SplitFed Learning framework.  
- **`losses` Folder**: Contains the training loss values reported during the training process and the `scores.json` file that tracks validation loss values.  
- **Malicious Client Logic**: Malicious nodes are configured to perform data poisoning attacks by sending randomized, harmful updates to the server.  

### `nodes` Folder

In our setup, a ***node*** refers to any entity in the distributed learning framework, either a client or a server.  

- **Flask-Based Node Communication**: Each node is implemented as a Flask server, enabling inter-node communication required for distributed learning.  
- **9 Nodes Configuration**: This setup includes 8 clients (3 malicious) and 1 central server. Each node is saved as a process in the `nodes` folder.  
- **Global Model Definition**: The model architecture is defined in the `global_var.py` file and can be modified to suit experimental needs.  

### `run.py` and `stop.py`  

- **`run.py`**: Initializes and begins the SplitFed Learning experiment, including setting up malicious nodes.  
- **`stop.py`**: Gracefully stops all running processes associated with the experiment.  

## Malicious Nodes  

The malicious nodes are configured to perform data poisoning attacks by:  
1. Sending harmful updates with randomized data during the training process.  
2. Submitting noisy gradients to the central server to hinder global model performance.  
This logic is implemented within the `Client` class, under the `attack` method.  

## Running the Experiment  

To replicate the attacked SplitFed Learning scenario:  

1. Install the required dependencies:  
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
The experiment evaluates the resilience of the Split Learning framework under data poisoning attacks.

- Loss values and model performance metrics are logged in the `losses` folder.
- The impact of malicious nodes can be assessed by analyzing the validation loss stored in `scores.json`.

This setup allows you to explore the vulnerabilities of Split Learning to data poisoning attacks and serves as a benchmark for evaluating potential countermeasures.

