# reinforcement-learning-for-per-flow-buffer-sizing
Contact: Maximilian Bachl

# Installation

## Downloading libtorch

* Download the [C++ version of libtorch for CPUs](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.5.1%2Bcpu.zip)
* Unpack it and move the contained directory ```libtorch``` to the root directory of this repository (the same directory in which there's the ```ns-allinone-3.30.1``` directory)

## Compiling

* ```cd ns-allinone-3.30.1/ns-3.30.1/```
* ```./waf clean```
* ```./waf configure```
* ```./waf build```

# Training an RL model

If the build finished successfully (this can take some time), you can train a model. 

Assuming you're in the ```ns-3.30.1``` folder, run 

    OMP_NUM_THREADS=1 LD_LIBRARY_PATH=../../libtorch/lib:$LD_LIBRARY_PATH ./waf -v --run "examples/traffic-control/rl"
    
# Evaluating

## Evaluating an RL model

To evaluate a set of trained weights run 

    OMP_NUM_THREADS=1 LD_LIBRARY_PATH=../../libtorch/lib:$LD_LIBRARY_PATH ./waf -v --run "examples/traffic-control/rl <path_to_weights>"
    
## Evaluating other AQM mechanisms

To evaluate, for example, FqCoDel, run

        OMP_NUM_THREADS=1 LD_LIBRARY_PATH=../../libtorch/lib:$LD_LIBRARY_PATH ./waf -v --run "examples/traffic-control/rl FqCoDelQueueDisc"
        
To run Fifo with a maximum queue size of 100, run

        OMP_NUM_THREADS=1 LD_LIBRARY_PATH=../../libtorch/lib:$LD_LIBRARY_PATH ./waf -v --run "examples/traffic-control/rl FifoQueueDisc 100"
        
# Plotting

## Plotting training runs

During training, various values are logged.

To plot them, run the following command:

    ./plot_metrics.py results/RLQueueDisc/logs/<name_of_the_file_to_be_plotted>

## Plotting evaluation results

The following commands are supposed to be run after an AQM mechanism was evaluated. 

To plot, for example, all queue traces produced by an evaluation of Fifo with queue size 1000, run the following command:

    ls -d $PWD/results/FifoQueueDisc/queueTraces/1000/*.plotme | xargs ./plot_something.py
    
To plot the behavior of an AQM mechanism and get performance metrics, run, for example, the following command:

    ./plot_df.py results/RLQueueDisc/logs/*
