# CPS Simulation with Agentic Capabilities: Independent Workstations (v1.0)

This project is a discrete-event simulation of a factory environment, built using the SimPy library in Python. It models a system of machines that operate as independent, parallel workstations, suitable for a job-shop manufacturing process where different tasks are performed concurrently. 
The core of this simulation is an autonomous agent that monitors the health and status of the machines, detects faults, and performs corrective actions to ensure the factory remains operational. The Agentic capabilities are just informational for now.


## Key Features

-   **Discrete-Event Simulation:** Leverages the SimPy framework to model the passage of time and concurrent processes.
-   **Independent Machine Schedules:** Each machine works through its own assigned "to-do list" of products, independent of other machines.
-   **Configurable via CSV:** The entire factory layout—including machine definitions, their connections, and their production schedules—is configured externally through simple CSV files, requiring no changes to the core Python code.
-   **Dynamic Fault Injection:** A `fault_injector` process randomly introduces realistic faults (sensor drift, stuck actuators, etc.) into the system to test its resilience.
-   **Autonomous Agent Monitoring:** A monitoring agent runs on a periodic cycle to check machine status, identify anomalies like low batteries or critical faults, and dispatch corrective actions.
-   **Centralized Logging:** All events from every component (machines, network, agent) are logged to a single `simulation.log` file with clear timestamps and component identifiers.

## System Architecture

The simulation is composed of several key classes and processes that interact to create the digital twin:

-   **`Machine`**: The core worker entity. It contains its own `Sensor` and `Actuator` components, manages its battery level, and processes a production schedule.
-   **`AgentAPI`**: A clean interface or "toolbox" that the agent uses to perceive and act upon the environment. It abstracts the underlying complexity of interacting with machine objects.
-   **`Network`**: A simple communication channel. In this version, it is used to send **informational messages** (e.g., "Product X is complete") from one machine to another, not to transfer physical products.
-   **`dummy_agent_process`**: The agent's "brain." It follows a simple loop: sleep for a set time, then wake up to check every machine and decide on actions.
-   **`fault_injector`**: An independent process that injects faults into sensors and actuators at random intervals.
-   **`setup_simulation_from_csv`**: The "factory builder" function that reads the `.csv` configuration files at the start and constructs the entire simulation environment.

## Example Scenario: "Artisan Furniture Co."

This project is perfectly suited to model a workshop where specialized machines work on different projects in parallel.

**The Setup:**
*   **`CNC-Cutter-01`**: A large CNC machine that cuts big wooden pieces, like tabletops and shelving units. It works completely independently.
*   **`Frame-Welder-01`**: A welding station that builds metal frames for coffee tables and chairs. It works on its own separate schedule.
*   **`CNC-Cutter-02`**: A smaller CNC machine that cuts the matching wooden tops for the coffee tables. To ensure the wood and metal parts for a project are ready around the same time, this machine sends a **notification** to `Frame-Welder-01` when it finishes a top. This allows the welder operator to know they can grab the finished top for final offline assembly.

This scenario highlights the model's key feature: machines working on their own schedules, with connections acting as simple notification triggers.

## Getting Started

Follow these steps to run the simulation.

### 1. Prerequisites

Ensure you have Python 3 installed. Then, install the necessary libraries:

```sh
pip install simpy pandas
