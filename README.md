# VarLogger: Lightweight Fault Detection and Diagnosis for IoT Sensor Nodes
This repository contains a lightweigth fault detection and diagnosis method which can be deployed on the microcontrollers to monitor them during runtime and thus, increase reliability of the IoT application at hand. Varlogger consists of the **node client** required to collect the data (*event traces*), as well as the **feature extraction** methods (*ST and EI*) to enable fault detection and diagnosis. 

This repository also contains the implementation of **SOTA** methods that we used to benchmark our performance.
<!-- This branch contains the code used to produce the results for *VarLogger: Anomaly Detection in IoT Sensor Nodes* publication. -->

The fault detection in VarLogger works in two parts: 
**a) Logging**: collecting the training data from the sensor nodes defining the correct behavior of the sensor nodes; 
**b) Training**: using the collected data to train lightwieght fault detection models based on extracted features.

## Node Client Library
This is the micropython library used **Logging**. VarLogger collects event traces from the sensor nodes which are further used for training and detection. Event trace is a time series data collected by monitoring the execution of code line by line running on the deployed sensor nodes. 

In order to collect the event trace from the sensor node, the developer needs to instrument the source code to have the logging commands after each line. The developer need to make use of the node client `varlogger.py`.

### Library placement
This library should be placed in the ```/lib``` folder along with other supporting flies and uploaded on the microcontroller along with the `main.py`
```
├── flash
│   ├── lib
│   │   ├── **
│   │   ├── varlogger.py
│   ├── main.py
│   ├── boot.py
````

### Library Usage
To add the logging command `log()` in the source code, one needs to make sure the `varlogger.py` is present in `lib` folder in the workspace so that it can be imported in `main.py`.

```python
import lib.varlogger as vl

vl.log(var=<variable name>, 
        fun=<function name>, 
        clas=<class name>, 
        th=<thread ID>)
```

`log()` command takes four input arguments. `var`: is the variable name of the user defined variable in the previous line of code; `fun`: is the name of the function to which the previous line of code belongs; `cls`: is the name of the class to which it belongs; `th`: is the thread id to which it belongs.

> Incase there is no valid input for any argument the default value is `0`

The `log` command should be added after every line of the code that contains user defined variable. This ensure that entire code is reachable. 

[Temp-Sensor](https://github.com/saurabh-2905/TempSensor/tree/main/transmitter) is one of the use cases, where VarLogger is integrated to detect anomales during runtime. Please refer to the source code to see usage of `vl.log()`

> **_NOTE:_** *We are currently working on the tool to automate the code instrumentation process. This tool will take the source code as input and will output the instrumented code where log statements are inserted.* 

## Feature Extractor
TODO

## Runtime Detection
TODO