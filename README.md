# Enhancing Split Learning with Sharded and Blockchain-Enabled SplitFed Approaches

This repository hosts the implementation of Sharded SplitFed Learning (SSFL) and Blockchain-enabled SplitFed Learning (BSFL), alongside the complete code for SplitFed Learning (SFL) and Split Learning (SL). To thoroughly evaluate our proposed methods, we conduct experiments under various conditions, including node counts of 9 and 36, in both normal and attacked scenarios. Each experiment is organized in a dedicated branch to facilitate reproducibility of our results.

## Branches
Each branch contains the implementation for a specific framework (SL, SFL, SSFL, or BSFL) and the corresponding steps required to run the experiments and reproduce the results presented in the paper.

### Branch Naming Convention
Branch names clearly indicate the specific experiment and its configuration. For instance, the branch `attack_blockchain_splitfed_36` contains the code and results for a BSFL experiment conducted with 36 nodes under a data poisoning attack. If a branch does not explicitly specify the number of nodes, it represents an experiment with 9 nodes.

### Branch: Plot
This branch aggregates the results from all experiments and presents them through multiple figures for comparison and visualization.
