# CPS Simulation with Agentic Capabilities: Independent Workstations (v1.5)

This project is a discrete-event simulation of a factory environment, built using the SimPy library in Python. It models a system of machines that operate as independent, parallel workstations, suitable for a job-shop manufacturing process where different tasks are performed concurrently. 
The core of this simulation is an autonomous agent that monitors the health and status of the machines, detects faults, and performs corrective actions to ensure the factory remains operational. The Agentic capabilities are just informational for now.

This builds on v1.0 and introduces Terminal UI, MockLLM calls via JSON


- ![alt text][arch]

[arch]: https://github.com/SabariNathanA/Simulated-Agent-CPS-v1/blob/v2-with-mock-LLM/arch9.2.png "Architecture Diagram"


## Example Scenario: Car Manufacturing Factory

This project is perfectly suited to model a car factory. Below is a schematic. 
- ![alt text][layout]

[layout]: https://github.com/SabariNathanA/Simulated-Agent-CPS-v1/blob/v2-with-mock-LLM/layout_bottleneck.drawio.png "Factory Layout"


## Getting Started

Follow these steps to run the simulation.

### 1. Prerequisites

Ensure you have Python 3 installed. Then, install the necessary libraries:

```sh
pip install simpy pandas
```

### 2. Configuration

1.  Use the main Python code modular_cps_v9a.py.
2.  In the same directory, create the two CSV configuration files machines_bottleneck.csv and schedules_bottleneck.csv.


### 3. Execution
Run the simulation from your terminal:
```sh
python modular_cps_v9a.py
```

### 4. Output
-   Console output:  You will see high-level status updates. You can pause the simulation by pressing p and resume by clicking r. 
-   simulation.log: A new file will be created in your directory. This file contains a detailed, timestamped log of every event that occurred during the simulation, providing a complete trace for analysis.
