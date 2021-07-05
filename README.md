# Partitioned scheduling on RT-FaaS

This repo contains the partitioning algorithm, the comparison simulator and the simulated results of the scientific paper titled '_Real-time task scheduling in a FaaS cloud_', presented in [IEEE CLOUD 2021](https://conferences.computer.org/cloud/2021/) 

### How to start the simulator

First, you need to fine tune the simulation parameters: 
  * TASK_COUNT (number of tasks 'n')
  * CPU_COUNT (number of processors 'm')
  * SIMULATION_COUNT (simulation repeat count with input 'n' and 'm')

To configure the parameters, just optn the file [simulation_for_paper.py](https://github.com/hsnlab/partitioned_scheduling_on_RT-FaaS/blob/main/cloud2021_paper_materials/simulation_for_paper.py) in directory [cloud2021_paper_materials](https://github.com/hsnlab/partitioned_scheduling_on_RT-FaaS/tree/main/cloud2021_paper_materials) and modify the variables at the begining of the source code.

The simulation results will be placed into _csv_ files in the same directory from where the simulator was executed.

**Run simulations:**
```
cd cloud2021_paper_materials
./simulation_for_paper.py
```

### Analize the simulation results

We have created jupyter notebooks to process the simulator output csv files and generate easily readable plots to conclude the results.
The notebook files are located in the directory [cloud2021_paper_materials](https://github.com/hsnlab/partitioned_scheduling_on_RT-FaaS/tree/main/cloud2021_paper_materials):
    * [Evaluation_heterogeneous_multicore_multinode_cluster.ipynb](https://github.com/hsnlab/partitioned_scheduling_on_RT-FaaS/blob/main/cloud2021_paper_materials/Evaluation_heterogeneous_multicore_multinode_cluster.ipynb)
    * [Evaluation_homogeneous_multicore_multinode_cluster.ipynb](https://github.com/hsnlab/partitioned_scheduling_on_RT-FaaS/blob/main/cloud2021_paper_materials/Evaluation_homogeneous_multicore_multinode_cluster.ipynb)

The simulated data from the paper are located in directory [cloud2021_paper_materials/dataset_paper](https://github.com/hsnlab/partitioned_scheduling_on_RT-FaaS/tree/main/cloud2021_paper_materials/dataset_paper).
