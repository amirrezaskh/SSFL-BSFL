# SplitFed Learning  
This branch implements the SplitFed Learning (SFL) algorithm with 35 clients and a central server, totaling 36 computing nodes. The following is an explanation of the algorithm and its implementation in our code.  

## Code Structure  

### `splitfed_learning` Folder  
This folder contains the specific code for implementing the SplitFed Learning framework. The framework is organized around two main classes: `server` and `client`. These classes represent the central server and the client entities in the SFL architecture, respectively.  

Additionally, the `losses` folder within this directory stores the training loss values reported by the central server during training. It also includes the `scores.json` file, which tracks the validation loss values throughout the training process.  

### `nodes` Folder  
In SplitFed Learning, a ***node*** is any entity in the distributed learning algorithm that can function as either a server or a client. For our implementation, nodes are designed as Flask servers to facilitate intercommunication between processes in a distributed environment.  

This folder contains the configuration for 36 nodes, with 35 acting as clients and 1 as the central server. The global model architecture used in this experiment is also stored in this folder and can be modified by updating the `global_var.py` file.  

### `run.py` and `stop.py`  
- **`run.py`**: Starts the experiment by spawning processes to initialize and run the nodes.  
- **`stop.py`**: Stops all running processes related to the experiment.  

## Running the Experiment  
To run the SplitFed Learning experiment, follow these steps:  

1. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
