# Parameters

## Experiment type
- `task`: Task type (`MNIST` or `expandedNARMA` or `memory` or `Lissajous` or `locomotion`)
- `experiment`: Experiment type

## Simulation settings
- `simTime`: Simulation time
- `timeStep`: Time step
- `tau`: Time scaling variable (ex. to define how many steps the input value changes)

## Brain setting
- `brain`: Brain type (`LIL` or `MLP` or `CNN` or `SWG` or `FL`)

## Body settings
- `shape`: Shape of MSDN (`DoubleCircles` or `MultiCircles` or `CaterpillarRobot`)
- `size`: Size of MSDN
- `nodes`: Number of mass points
- `neighbours`: Number of neighbours to which a spring connection is made (if shape is `CaterpillarRobot`)

## Training settings
- `batchSize`: Batch size
- `epoch`: Training epoch
- `optim`: Optimizer (`Adam` or `SGD` or `RMSprop`)
- `updateTimeScale`: Number of steps to update parameters

## Learning rates
- `lr_spring`: Learning rate of spring constant of MSDN
- `lr_damping`: Learning rate of damping coefficient of MSDN
- `lr_restlength`: Learning rate of rest length of MSDN
- `lr_brain`: Learning rate of FNN parameters
- `lr_amplitude`: Learning rate of amplitude of SWG
- `lr_phase`: Learning rate of phase of SWG

## Others
- `pixel`: Number of pixels on each side of the MNIST image
- `fixedImages`: Whether to train and test only on specific MNIST images (if -1, use all)
- `dataSize`: Number of data to use for training (if -1, use all)
- `testDataSize`: Number of data to use for testing (if -1, use all)
- `char`: Character to use for training and testing (if -1, use all)
- `deltaScale`: Number of pixels on each side of label map
- `delay`: Step delay in timteseries emulation tasks
- `inputScale`: Input value scaling
- `cimulative`: Whether to set the target output as the cumulative value of past inputs in memory task
- `wind`: Whether to introduce wind in the experiment
- `lissajous1`: First target Lissajous curve's parameters
- `lissajous2`: Second target Lissajous curve's parameters
- `noise`: Standard deviation of noise added to feedback information
- `generatorDist`: Distribution of springs modulated by SWG (if shape is `DoubleCircles`, `all` or `ex` or `mid` or `in`)
- `feedbackSprings`: Distribution of springs used for feedback (if shape is `DoubleCircles`, `all` or `ex` or `mid` or `in` or `exmid` or `exin` or `midin`; if shape is `CaterpillarRobot`, `all` or `half` or `quater`)
- `reservoirStateType`: Feedback information (`springLength` or `massPosition` or `massSpeed` or `massAcceleration`)
