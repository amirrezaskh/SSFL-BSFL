# Split Learning
This branch implements the split learning algorithm with 35 clients and a central server, 36 computing nodes in total. The following is an explanation of the algorithm and how it is implemented in our code.

## Code Structure

### `split_learning` Folder

This folder contains the specific code for achieving split learning. We define two classes named `server` and `client` where they represent the server and client entities in split learning.

Inside this folder, you can also find the `losses` folder. The `losses` folder contains the training loss value reported by the central server during the training process as well as the `scores.json` file that contains the validaiton loss values.

### `nodes` Folder

In our setup, we define a ***node*** as an entity in a distributed learning algorithm which can serve as a server or a client. We developed nodes as Flask servers to enable intercommunication between processes to implement distributed learning. In this branch, we experiment with split learning with 36 nodes, 35 clients and 1 server, where the nodes are saved in the `nodes` folder. 

This folder also contains the global model architecture selected for this experiment. The model can be easily modified through changes to the `global_var.py` file.

### `run.py` and `stop.py`
These two files simply begin and end the experiment after being run. 

## Running the experiment.
To run this experiment, first install the dependencies:
```
pip install requirements.txt
```
Then run the following command:
```
python3 run.py
```
which will spawn processes in the background that will run the experiment. The processes will exit as they complete the experiment, nevertheless, if you want to stop the code from running, you can simply run the following command:
```
python3 stop.py
```
