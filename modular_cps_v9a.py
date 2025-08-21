# v1.8 - Final Demo Version
# Re-introduces the 'dummy_agent_process' to allow for A/B comparison.
# Provides a clear switch in the main block to choose between the simple and smart agent.
# Includes non-flickering pause and the informative UI.

import simpy
import random
import pandas as pd
import logging
import os
from copy import deepcopy
import json
import threading
import keyboard
import time

# --- Simulation State & Keyboard Listener (Unchanged) ---
class SimulationState:
    def __init__(self):
        self.is_paused = False

sim_state = SimulationState()

def keyboard_listener(state):
    while True:
        try:
            keyboard.wait('p')
            state.is_paused = True
            keyboard.wait('r')
            state.is_paused = False
        except:
            break

# --- LiveDashboard Class (Unchanged) ---
class LiveDashboard:
    def __init__(self, env, kpi_tracker, machines):
        self.env = env
        self.kpi_tracker = kpi_tracker
        self.machines = machines
        self.last_machine_statuses = {m_id: "" for m_id in machines}

    def get_status_and_details(self, machine):
        if machine.actuators[0].fault_type == 'stuck':
            return "STUCK", f"FAULT: {machine.actuators[0].actuator_type} stuck"
        for component in machine.actuators + machine.sensors:
            if component.fault_type:
                comp_type = "Actuator" if isinstance(component, Actuator) else "Sensor"
                return "FAULT", f"FAULT: {comp_type} {component.fault_type}"
        if machine.is_processing:
            return "PROCESSING", f"Product: {machine.current_product_id}"
        return "IDLE", "--"

    def display(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        if sim_state.is_paused:
            print("--- SIMULATION PAUSED (Press 'r' to Resume) ---")
        else:
            print("--- CPS Digital Twin v1.8 --- [LIVE] (Press 'p' to Pause) ---")
        print(f"Simulation Time: {self.env.now:.2f}s\n")
        active_faults = len(self.kpi_tracker.open_faults)
        avg_mttr = sum(self.kpi_tracker.repair_times) / len(self.kpi_tracker.repair_times) if self.kpi_tracker.repair_times else 0.0
        print("[ KPIs ]")
        print(f"> Total Throughput: {self.kpi_tracker.throughput} parts")
        print(f"> Mean Time To Repair (MTTR): {avg_mttr:.2f}s")
        print(f"> Active Faults: {active_faults}\n")
        print("[ MACHINE STATUS ]")
        print(f"{'ID':<25} {'STATUS':<15} {'BATTERY':<10} {'COMPLETED':<8} {'QUEUE':<8} {'DETAILS'}")
        print("-" * 85)
        current_statuses = {}
        for machine_id, machine in self.machines.items():
            status, details = self.get_status_and_details(machine)
            current_statuses[machine_id] = status
            highlight_char = ">>" if self.last_machine_statuses.get(machine_id) != status else "  "
            battery_str = f"{machine.battery_level:.1f}%"
            parts_str = str(machine.parts_produced)
            queue_str = str(len(machine.production_schedule))
            print(f"{highlight_char} {machine_id:<22} {status:<15} {battery_str:<10} {parts_str:<8} {queue_str:<8} {details}")
        self.last_machine_statuses = current_statuses
        print("\n--- Press Ctrl+C to stop the simulation ---")

# --- Simulation Parameters & Logging (Unchanged) ---
MEAN_TIME_BETWEEN_FAULTS = 25.0
SENSOR_FAULT_TYPES = ['drift', 'stuck', 'offset']
ACTUATOR_FAULT_TYPES = ['slow_response', 'stuck']
BATTERY_DEPLETION_RATE = 0.1
LOW_BATTERY_THRESHOLD = 20.0
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(component)s) - %(message)s', filename='simulation.log', filemode='w')

# --- All Classes (Sensor, Actuator, KPI_Tracker, Network, Machine) are Unchanged ---
# (Omitted for brevity, but they are the same as the complete versions from the last correct response)
class Sensor:
    def __init__(self, env, machine_name, sensor_type, true_value_func):
        self.env = env
        self.machine_name = machine_name
        self.sensor_type = sensor_type
        self.true_value_func = true_value_func
        self.fault_type = None
        self.fault_param = 0.0
    def read_value(self):
        true_value = self.true_value_func()
        if self.fault_type is None: measured_value = true_value
        elif self.fault_type == 'drift':
            self.fault_param += (random.random() - 0.5) * 0.1
            measured_value = true_value + self.fault_param
        elif self.fault_type == 'stuck': measured_value = self.fault_param
        elif self.fault_type == 'offset': measured_value = true_value + self.fault_param
        return measured_value
    def induce_fault(self, fault_type):
        self.fault_type = fault_type
        true_value = self.true_value_func()
        if self.fault_type == 'drift': self.fault_param = 0.0
        elif self.fault_type == 'stuck': self.fault_param = true_value + (random.random() - 0.5) * 5
        elif self.fault_type == 'offset': self.fault_param = (random.random() * 10) - 5
        logging.error(f"FAULT INDUCED on {self.machine_name}-{self.sensor_type}: Type: {self.fault_type}", extra={'component': 'FaultInjector'})
    def clear_fault(self):
        logging.info(f"Corrective action: Clearing fault on {self.machine_name}-{self.sensor_type}", extra={'component': 'System'})
        self.fault_type = None
        self.fault_param = 0.0
class Actuator:
    def __init__(self, env, machine_name, actuator_type):
        self.env = env
        self.machine_name = machine_name
        self.actuator_type = actuator_type
        self.fault_type = None
        self.fault_param = 0.0
        self.stuck_process = None
        self.base_action_time = 0.1
    def perform_action(self, action):
        if self.fault_type == 'stuck':
            try:
                logging.critical(f"CRITICAL FAULT {self.machine_name}: Actuator '{self.actuator_type}' is STUCK trying to '{action}'.", extra={'component': 'Actuator'})
                self.stuck_process = self.env.process(self._do_stuck_action())
                yield self.stuck_process
            except simpy.Interrupt:
                logging.warning(f"Actuator on {self.machine_name} was forcefully interrupted by the agent.", extra={'component': 'Actuator'})
            return
        action_time = self.base_action_time
        if self.fault_type == 'slow_response':
            action_time += self.fault_param
            logging.warning(f"{self.machine_name}: Actuator '{self.actuator_type}' is slow. Action '{action}' taking {action_time:.2f}s.", extra={'component': 'Actuator'})
        else:
            logging.info(f"{self.machine_name}: Actuator '{self.actuator_type}' performing action: '{action}'.", extra={'component': 'Actuator'})
        yield self.env.timeout(action_time)
    def _do_stuck_action(self):
        yield self.env.timeout(float('inf'))
    def induce_fault(self, fault_type):
        self.fault_type = fault_type
        if self.fault_type == 'slow_response': self.fault_param = random.uniform(0.5, 2.0)
        logging.error(f"FAULT INDUCED on {self.machine_name}-{self.actuator_type}: Type: {self.fault_type}", extra={'component': 'FaultInjector'})
    def clear_fault(self):
        logging.info(f"Corrective action: Clearing fault on {self.machine_name}-{self.actuator_type}", extra={'component': 'System'})
        self.fault_type = None
        self.fault_param = 0.0
class KPI_Tracker:
    def __init__(self, env):
        self.env = env
        self.throughput = 0
        self.machine_states = {}
        self.open_faults = {}
        self.repair_times = []
    def initialize_machine_states(self, machines):
        for machine_id in machines:
            self.machine_states[machine_id] = {'total_processing_time': 0.0, 'last_change_time': 0.0, 'is_processing': False}
    def track_production(self):
        self.throughput += 1
    def track_machine_state_change(self, machine_id, is_processing):
        now = self.env.now
        state_data = self.machine_states.get(machine_id)
        if not state_data: return
        if state_data['is_processing']:
            state_data['total_processing_time'] += now - state_data['last_change_time']
        state_data['last_change_time'] = now
        state_data['is_processing'] = is_processing
    def track_fault_start(self, machine_id, component_name):
        fault_key = f"{machine_id}-{component_name}"
        self.open_faults[fault_key] = self.env.now
        logging.info(f"KPI: Fault started for {fault_key} at T={self.env.now:.2f}", extra={'component': 'KPI_Tracker'})
    def track_fault_end(self, machine_id, component_name):
        fault_key = f"{machine_id}-{component_name}"
        if fault_key in self.open_faults:
            start_time = self.open_faults.pop(fault_key)
            duration = self.env.now - start_time
            self.repair_times.append(duration)
            logging.info(f"KPI: Fault ended for {fault_key} at T={self.env.now:.2f}. Repair time: {duration:.2f}", extra={'component': 'KPI_Tracker'})
    def generate_report(self):
        print("\n--- KPI Summary Report ---")
        print(f"Total Throughput: {self.throughput} parts")
        for machine_id, data in self.machine_states.items():
            if data['is_processing']:
                data['total_processing_time'] += self.env.now - data['last_change_time']
        total_utilization = 0
        num_machines = len(self.machine_states)
        if num_machines > 0:
            for machine_id, data in self.machine_states.items():
                utilization_percent = (data['total_processing_time'] / self.env.now) * 100 if self.env.now > 0 else 0
                print(f"  - Machine '{machine_id}' Utilization: {utilization_percent:.2f}%")
                total_utilization += utilization_percent
            avg_utilization = total_utilization / num_machines
            print(f"Average Machine Utilization: {avg_utilization:.2f}%")
        if self.repair_times:
            avg_mttr = sum(self.repair_times) / len(self.repair_times)
            print(f"Mean Time To Repair (MTTR): {avg_mttr:.2f} seconds (from {len(self.repair_times)} repairs)")
        else:
            print("Mean Time To Repair (MTTR): N/A (No repairs were completed)")
        if self.open_faults:
            print(f"Warning: {len(self.open_faults)} faults were still unresolved at the end of the simulation.")
        print("--- End of Report ---\n")
class Network:
    def __init__(self, env):
        self.env = env
    def send_message(self, sender, recipient, message):
        latency = abs(random.normalvariate(0.05, 0.01))
        yield self.env.timeout(latency)
        if random.random() < 0.1:
            logging.warning(f"Network: Message from {sender} to {recipient} LOST.", extra={'component': 'Network'})
        else:
            logging.info(f"Network: Message from {sender} to {recipient} delivered: '{message}'", extra={'component': 'Network'})
class Machine:
    def __init__(self, env, machine_id, production_schedule, network, kpi_tracker, next_machine=None):
        self.env = env
        self.id = machine_id
        self.original_schedule = deepcopy(production_schedule)
        self.production_schedule = deepcopy(production_schedule)
        self.network = network
        self.next_machine = next_machine
        self.machine_resource = simpy.Resource(env, capacity=1)
        self.battery_level = 100.0
        self.is_processing = False
        self.parts_produced = 0
        self.current_product_id = None
        self.process_time_for_interrupt = 0
        self.sensors = [Sensor(self.env, self.id, "Temperature", lambda: 90.0+(5 if self.is_processing else -5)), Sensor(self.env, self.id, "Power", lambda: 5.0 if self.is_processing else 0.5)]
        self.actuators = [Actuator(self.env, self.id, "PartFeeder")]
        self.part_feeder_actuator = self.actuators[0]
        self.kpi_tracker = kpi_tracker
        self.env.process(self.monitor_battery())
        self.env.process(self.read_sensors_periodically())
        self.production_process = self.env.process(self.run_production())
    def run_production(self):
        try:
            while len(self.production_schedule) > 0:
                product_id, process_time = self.production_schedule.pop(0)
                self.process_time_for_interrupt = process_time
                with self.machine_resource.request() as request:
                    yield request
                    self.current_product_id = product_id
                    logging.info(f"{self.id}: Starting production of {product_id}.", extra={'component': self.id})
                    self.kpi_tracker.track_machine_state_change(self.id, True)
                    self.is_processing = True
                    yield self.env.process(self.part_feeder_actuator.perform_action(f"Feed part for {product_id}"))
                    yield self.env.timeout(process_time)
                    self.kpi_tracker.track_production()
                    self.parts_produced += 1
                    self.kpi_tracker.track_machine_state_change(self.id, False)
                    self.is_processing = False
                    logging.info(f"{self.id}: Finished production of {product_id}.", extra={'component': self.id})
                    if self.next_machine:
                        self.env.process(self.network.send_message(self.id, self.next_machine.id, f"Product {product_id} finished."))
            self.current_product_id = None
        except simpy.Interrupt:
            logging.warning(f"INTERRUPTED: Production process for {self.id} was interrupted by agent.", extra={'component': self.id})
            if self.current_product_id:
                logging.info(f"Restoring product {self.current_product_id} to schedule for {self.id}.", extra={'component': self.id})
                self.production_schedule.insert(0, (self.current_product_id, self.process_time_for_interrupt))
            if self.is_processing:
                 self.kpi_tracker.track_machine_state_change(self.id, False)
            self.is_processing = False
            self.current_product_id = None
    def monitor_battery(self):
        while self.battery_level > 0:
            self.battery_level -= BATTERY_DEPLETION_RATE if self.is_processing else BATTERY_DEPLETION_RATE/5
            if self.battery_level < LOW_BATTERY_THRESHOLD: logging.warning(f"{self.id}: LOW BATTERY! Level: {self.battery_level:.2f}%", extra={'component': self.id})
            yield self.env.timeout(1)
        logging.critical(f"{self.id}: BATTERY DEAD.", extra={'component': self.id})
    def read_sensors_periodically(self):
        while self.battery_level > 0:
            yield self.env.timeout(5)
class AgentAPI:
    def __init__(self, env, machines,kpi_tracker):
        self.env = env
        self.machines = machines
        self.kpi_tracker = kpi_tracker
        logging.info("AgentAPI initialized.", extra={'component': 'AgentAPI'})
    def list_all_machines(self): return list(self.machines.keys())
    def get_machine_status(self, machine_id):
        if machine_id not in self.machines: return None
        m = self.machines[machine_id]
        is_stuck = m.actuators[0].fault_type == 'stuck'
        return {"is_processing": m.is_processing, "battery_level": m.battery_level, "parts_produced": m.parts_produced, "current_product": m.current_product_id, "is_stuck": is_stuck}
    def get_component_faults(self, machine_id):
        if machine_id not in self.machines: return []
        m = self.machines[machine_id]
        active_faults = []
        for sensor in m.sensors:
            if sensor.fault_type:
                active_faults.append({'component_type': 'sensor', 'component_name': sensor.sensor_type, 'fault_type': sensor.fault_type})
        for actuator in m.actuators:
            if actuator.fault_type and actuator.fault_type != 'stuck':
                active_faults.append({'component_type': 'actuator', 'component_name': actuator.actuator_type, 'fault_type': actuator.fault_type})
        return active_faults
    def read_all_sensor_values(self, machine_id):
        if machine_id not in self.machines: return None
        return {s.sensor_type: s.read_value() for s in self.machines[machine_id].sensors}
    def get_production_schedule(self, machine_id):
        if machine_id not in self.machines: return None
        return self.machines[machine_id].production_schedule
    def update_production_schedule(self, machine_id, new_schedule):
        if machine_id not in self.machines: return False
        logging.info(f"AGENT ACTION: Updating schedule for {machine_id}", extra={'component': 'AgentAPI'})
        self.machines[machine_id].production_schedule = new_schedule
        return True
    def recalibrate_component(self, machine_id, component_type, component_name):
        if machine_id not in self.machines: return False
        m = self.machines[machine_id]
        components = m.sensors if component_type == 'sensor' else m.actuators
        comp_name_attr = 'sensor_type' if component_type == 'sensor' else 'actuator_type'
        for comp in components:
            if getattr(comp, comp_name_attr) == component_name:
                logging.info(f"AGENT ACTION: Recalibrating {component_name} on {machine_id}", extra={'component': 'AgentAPI'})
                comp.clear_fault()
                self.kpi_tracker.track_fault_end(machine_id, component_name)
                return True
        return False
    def dispatch_battery_replacement(self, machine_id):
        if machine_id not in self.machines: return False
        logging.info(f"AGENT ACTION: Dispatching battery replacement for {machine_id}", extra={'component': 'AgentAPI'})
        self.env.process(self._battery_replacement_process(machine_id))
        return True
    def _battery_replacement_process(self, machine_id):
        yield self.env.timeout(10.0)
        logging.info(f"Corrective action: Battery for {machine_id} has been replaced.", extra={'component': 'System'})
        if machine_id in self.machines: self.machines[machine_id].battery_level = 100.0
    def reboot_machine_process(self, machine_id):
        if machine_id not in self.machines: return False
        m = self.machines[machine_id]
        logging.info(f"AGENT ACTION: Rebooting production process for {machine_id}", extra={'component': 'AgentAPI'})
        for actuator in m.actuators:
            if actuator.stuck_process and actuator.stuck_process.is_alive:
                actuator.stuck_process.interrupt()
                actuator.stuck_process = None
        if m.production_process.is_alive:
            m.production_process.interrupt()
        m.production_process = self.env.process(m.run_production())
        return True

# --- fault_injector and setup_simulation_from_csv (Unchanged) ---
def fault_injector(env, all_components, kpi_tracker):
    while True:
        yield env.timeout(random.expovariate(1.0 / MEAN_TIME_BETWEEN_FAULTS))
        target_component = random.choice(all_components)
        if isinstance(target_component, Sensor):
            fault_type = random.choice(SENSOR_FAULT_TYPES)
            target_component.induce_fault(fault_type)
            kpi_tracker.track_fault_start(target_component.machine_name, target_component.sensor_type)
        elif isinstance(target_component, Actuator):
            fault_type = random.choice(ACTUATOR_FAULT_TYPES)
            target_component.induce_fault(fault_type)
            kpi_tracker.track_fault_start(target_component.machine_name, target_component.actuator_type)
def setup_simulation_from_csv(env, machines_csv, schedules_csv, kpi_tracker):
    machines_df = pd.read_csv(machines_csv); schedules_df = pd.read_csv(schedules_csv)
    schedules = (schedules_df.groupby('schedule_id')[['product_id', 'process_time']].apply(lambda g: list(g.itertuples(index=False, name=None))).to_dict())
    network = Network(env); machines = {}; all_sensors = []; all_actuators = []
    for _, row in machines_df.iterrows():
        machine_id = row['machine_id']; schedule_id = row['schedule_id']
        if schedule_id not in schedules: continue
        machine = Machine(env, machine_id, schedules[schedule_id], network, kpi_tracker)
        machines[machine_id] = machine; all_sensors.extend(machine.sensors); all_actuators.extend(machine.actuators)
        logging.info(f"Created machine: {machine_id}", extra={'component': 'Setup'})
    kpi_tracker.initialize_machine_states(machines.keys())
    for _, row in machines_df.iterrows():
        machine_id = row['machine_id']; next_machine_id = row['next_machine_id']
        if pd.notna(next_machine_id) and machine_id in machines and next_machine_id in machines:
            machines[machine_id].next_machine = machines[next_machine_id]
            logging.info(f"Linked {machine_id} -> {next_machine_id}", extra={'component': 'Setup'})
    return machines, all_sensors + all_actuators

# --- AGENT LOGIC SECTION (Both agents are now available) ---
def mock_llm_think(system_state_prompt):
    """
    A reliable, fast, "fake brain" that mimics an LLM's decision process for the simple agent.
    """
    if "is_stuck': True" in system_state_prompt:
        for line in system_state_prompt.split('\n'):
            if "is_stuck': True" in line:
                machine_id = line.split(':')[0].strip()
                command = {
                    "action": "recalibrate_and_reboot",
                    "params": { "machine_id": machine_id, "component_type": "actuator", "component_name": "PartFeeder" }
                }
                return json.dumps(command)
    return json.dumps({"action": "do_nothing", "params": {}})

def dummy_agent_process(env, api):
    """The baseline reactive agent. Only handles 'stuck' faults using the mock LLM logic."""
    logging.info("DUMMY Agent process started (Handles STUCK faults only).", extra={'component': 'DummyAgent'})
    while True:
        yield env.timeout(15)
        all_machines = api.list_all_machines()
        system_state_lines = []
        for machine_id in all_machines:
            status = api.get_machine_status(machine_id)
            if status:
                system_state_lines.append(f"{machine_id}: {status}")
        system_state_prompt = "\n".join(system_state_lines)
        llm_response_str = mock_llm_think(system_state_prompt)
        try:
            command = json.loads(llm_response_str)
            action = command.get("action")
            if action == "recalibrate_and_reboot":
                params = command.get("params", {})
                machine_id = params.get("machine_id")
                logging.warning(f"DummyAgent detected {machine_id} is STUCK! Executing recovery.", extra={'component': 'DummyAgent'})
                api.recalibrate_component(machine_id, params.get("component_type"), params.get("component_name"))
                api.reboot_machine_process(machine_id)
        except (json.JSONDecodeError, AttributeError):
            pass
        for machine_id in all_machines:
            status = api.get_machine_status(machine_id)
            if status and status['battery_level'] < LOW_BATTERY_THRESHOLD + 5:
                api.dispatch_battery_replacement(machine_id)

def smarter_agent_process(env, api):
    """A more advanced agent that handles all fault types."""
    logging.info("SMARTER Agent process started (Handles ALL faults).", extra={'component': 'SmarterAgent'})
    while True:
        yield env.timeout(15)
        all_machines = api.list_all_machines()
        for machine_id in all_machines:
            status = api.get_machine_status(machine_id)
            if not status: continue
            if status['battery_level'] < LOW_BATTERY_THRESHOLD + 5:
                api.dispatch_battery_replacement(machine_id)
            if status['is_stuck']:
                logging.warning(f"SmarterAgent detected {machine_id} is CRITICALLY STUCK!", extra={'component': 'SmarterAgent'})
                api.recalibrate_component(machine_id, 'actuator', 'PartFeeder')
                api.reboot_machine_process(machine_id)
                continue
            other_faults = api.get_component_faults(machine_id)
            if other_faults:
                for fault in other_faults:
                    logging.warning(f"SmarterAgent detected soft fault on {machine_id}: {fault}", extra={'component': 'SmarterAgent'})
                    api.recalibrate_component(machine_id, fault['component_type'], fault['component_name'])

# --- FINAL MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Starting CPS Simulation v1.8 (Final Demo Version) ---")
    REPRODUCIBILITY_SEED = 25
    random.seed(REPRODUCIBILITY_SEED)
    
    env = simpy.Environment()

    kpi_tracker = KPI_Tracker(env)
    machines, all_components = setup_simulation_from_csv(env, 'machines_bottleneck.csv', 'schedules_bottleneck.csv', kpi_tracker)
    
    simulation_duration = 300

    if machines:
        agent_api = AgentAPI(env, machines, kpi_tracker)
        env.process(fault_injector(env, all_components, kpi_tracker))
        
        # --- CHOOSE YOUR AGENT ---
        # To run with NO agent, comment out both lines.
        # To run with the SIMPLE agent, comment out the "smarter" line.
        # To run with the ADVANCED agent, comment out the "dummy" line.

        # env.process(dummy_agent_process(env, agent_api))
        env.process(smarter_agent_process(env, agent_api))
        
        listener_thread = threading.Thread(target=keyboard_listener, args=(sim_state,), daemon=True)
        listener_thread.start()

        live_dashboard = LiveDashboard(env, kpi_tracker, machines)
        
        print("--- Simulation starting. Press 'p' to pause, 'r' to resume. ---")
        step_duration = 1.0
        real_time_per_step = 0.1 #10 sim seconds per real second

        try:
            was_paused = False
            while env.peek() < simulation_duration:
                if sim_state.is_paused:
                    if not was_paused:
                        live_dashboard.display()
                        was_paused = True
                    time.sleep(0.1)
                    continue
                if was_paused:
                    was_paused = False
                
                env.run(until=env.now + step_duration)
                live_dashboard.display()
                time.sleep(real_time_per_step)
        except KeyboardInterrupt:
            print("\n--- Simulation stopped by user (Ctrl+C). ---")
        
    print("\n--- Simulation Finished ---")
    live_dashboard.display()
    kpi_tracker.generate_report()