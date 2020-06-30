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

## Training an RL model

If the build finished successfully (this can take some time), you can train a model. 

Assuming you're in the ```ns-3.30.1``` folder, run 

    OMP_NUM_THREADS=1 LD_LIBRARY_PATH=../../libtorch/lib:$LD_LIBRARY_PATH ./waf -v --run "examples/traffic-control/rl"
    
## Evaluating an RL model

To evaluate a set of trained weights run 

    OMP_NUM_THREADS=1 LD_LIBRARY_PATH=../../libtorch/lib:$LD_LIBRARY_PATH ./waf -v --run "examples/traffic-control/rl <path_to_weights>"
    
## Evaluating other AQM mechanisms

To evaluate, for example, FqCoDel, run

        OMP_NUM_THREADS=1 LD_LIBRARY_PATH=../../libtorch/lib:$LD_LIBRARY_PATH ./waf -v --run "examples/traffic-control/rl FqCoDelQueueDisc"
        
To run Fifo with a maximum queue size of 100, run

        OMP_NUM_THREADS=1 LD_LIBRARY_PATH=../../libtorch/lib:$LD_LIBRARY_PATH ./waf -v --run "examples/traffic-control/rl FifoQueueDisc 100"
        
## Plotting

### Plotting training runs

### Plotting evaluation results
