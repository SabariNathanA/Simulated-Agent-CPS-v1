    import simpy
import random
import pandas as pd
import logging
import os
from copy import deepcopy

#---Changes----
# clear_faults
# dummy agent class
# all agentic see and tools are in one class

# --- Simulation Parameters ---
MEAN_TIME_BETWEEN_FAULTS = 40.0
SENSOR_FAULT_TYPES = ['drift', 'stuck', 'offset']
ACTUATOR_FAULT_TYPES = ['slow_response', 'stuck']
BATTERY_DEPLETION_RATE = 0.1
LOW_BATTERY_THRESHOLD = 20.0
NETWORK_PACKET_LOSS_CHANCE = 0.1
NETWORK_LATENCY_MEAN = 0.05
NETWORK_LATENCY_STDDEV = 0.01
BATTERY_REPLACEMENT_TIME = 10.0

# --- Setup Centralized Logging ---
# CORRECTED: Using a custom 'component' field instead of the reserved 'processName'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(component)s) - %(message)s',
    filename='simulation.log',
    filemode='w'
)

# --- MODIFIED Sensor Class ---
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
        logging.info(f"{self.machine_name}: {self.sensor_type} reading: {measured_value:.2f} (True: {true_value:.2f}, Fault: {self.fault_type})", extra={'component': 'Sensor'})
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

# --- MODIFIED Actuator Class ---
class Actuator:
    def __init__(self, env, machine_name, actuator_type):
        self.env = env
        self.machine_name = machine_name
        self.actuator_type = actuator_type
        self.fault_type = None
        self.fault_param = 0.0
        self.base_action_time = 0.1

    def perform_action(self, action):
        if self.fault_type == 'stuck':
            logging.critical(f"CRITICAL FAULT 🛑 {self.machine_name}: Actuator '{self.actuator_type}' is STUCK trying to '{action}'.", extra={'component': 'Actuator'})
            yield self.env.timeout(float('inf'))
            return
        action_time = self.base_action_time
        if self.fault_type == 'slow_response':
            action_time += self.fault_param
            logging.warning(f"⚠️ {self.machine_name}: Actuator '{self.actuator_type}' is slow. Action '{action}' taking {action_time:.2f}s.", extra={'component': 'Actuator'})
        else:
            logging.info(f"⚡ {self.machine_name}: Actuator '{self.actuator_type}' performing action: '{action}'.", extra={'component': 'Actuator'})
        yield self.env.timeout(action_time)

    def induce_fault(self, fault_type):
        self.fault_type = fault_type
        if self.fault_type == 'slow_response': self.fault_param = random.uniform(0.5, 2.0)
        logging.error(f"FAULT INDUCED on {self.machine_name}-{self.actuator_type}: Type: {self.fault_type}", extra={'component': 'FaultInjector'})

    def clear_fault(self):
        logging.info(f"Corrective action: Clearing fault on {self.machine_name}-{self.actuator_type}", extra={'component': 'System'})
        self.fault_type = None
        self.fault_param = 0.0

class Network:
    def __init__(self, env):
        self.env = env
    def send_message(self, sender, recipient, message):
        latency = abs(random.normalvariate(NETWORK_LATENCY_MEAN, NETWORK_LATENCY_STDDEV))
        yield self.env.timeout(latency)
        if random.random() < NETWORK_PACKET_LOSS_CHANCE:
            logging.warning(f"❌ Network: Message from {sender} to {recipient} LOST.", extra={'component': 'Network'})
        else:
            logging.info(f"✉️ Network: Message from {sender} to {recipient} delivered: '{message}'", extra={'component': 'Network'})

# --- MODIFIED Machine Class ---
class Machine:
    def __init__(self, env, machine_id, production_schedule, network, next_machine=None):
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
        self.sensors = [Sensor(self.env, self.id, "Temperature", lambda: 90.0+(5 if self.is_processing else -5)), Sensor(self.env, self.id, "Power", lambda: 5.0 if self.is_processing else 0.5)]
        self.actuators = [Actuator(self.env, self.id, "PartFeeder")]
        self.part_feeder_actuator = self.actuators[0]
        
        self.env.process(self.monitor_battery())
        self.env.process(self.read_sensors_periodically())
        self.production_process = self.env.process(self.run_production())

    def run_production(self):
        try:
            while len(self.production_schedule) > 0:
                product_id, process_time = self.production_schedule.pop(0)
                with self.machine_resource.request() as request:
                    yield request
                    self.current_product_id = product_id
                    logging.info(f"▶️ {self.id}: Starting production of {product_id}.", extra={'component': self.id})
                    self.is_processing = True
                    yield self.env.process(self.part_feeder_actuator.perform_action(f"Feed part for {product_id}"))
                    yield self.env.timeout(process_time)
                    self.parts_produced += 1
                    self.is_processing = False
                    logging.info(f"✅ {self.id}: Finished production of {product_id}.", extra={'component': self.id})
                    if self.next_machine:
                        self.env.process(self.network.send_message(self.id, self.next_machine.id, f"Product {product_id} finished."))
            self.current_product_id = None
        except simpy.Interrupt:
            logging.warning(f"INTERRUPTED: Production process for {self.id} was interrupted by agent.", extra={'component': self.id})
            self.is_processing = False
            self.current_product_id = None

    def monitor_battery(self):
        while self.battery_level > 0:
            self.battery_level -= BATTERY_DEPLETION_RATE if self.is_processing else BATTERY_DEPLETION_RATE/5
            if self.battery_level < LOW_BATTERY_THRESHOLD: logging.warning(f"⚠️ {self.id}: LOW BATTERY! Level: {self.battery_level:.2f}%", extra={'component': self.id})
            yield self.env.timeout(1)
        logging.critical(f"🛑 {self.id}: BATTERY DEAD.", extra={'component': self.id})

    def read_sensors_periodically(self):
        while self.battery_level > 0:
            yield self.env.timeout(5)
            # No need to log here, read_value already logs

# --- NEW AgentAPI Class ---
class AgentAPI:
    def __init__(self, env, machines):
        self.env = env
        self.machines = machines
        # CORRECTED: Use 'component'
        logging.info("AgentAPI initialized.", extra={'component': 'AgentAPI'})

    # ... Perception Tools ...
    def list_all_machines(self): return list(self.machines.keys())
    def get_machine_status(self, machine_id):
        if machine_id not in self.machines: return None
        m = self.machines[machine_id]
        is_stuck = m.actuators[0].fault_type == 'stuck'
        return {"is_processing": m.is_processing, "battery_level": m.battery_level, "parts_produced": m.parts_produced, "current_product": m.current_product_id, "is_stuck": is_stuck}
    def read_all_sensor_values(self, machine_id):
        if machine_id not in self.machines: return None
        return {s.sensor_type: s.read_value() for s in self.machines[machine_id].sensors}
    def get_production_schedule(self, machine_id):
        if machine_id not in self.machines: return None
        return self.machines[machine_id].production_schedule
    
    # --- Action Tools ---
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
                return True
        return False
    def dispatch_battery_replacement(self, machine_id):
        if machine_id not in self.machines: return False
        logging.info(f"AGENT ACTION: Dispatching battery replacement for {machine_id}", extra={'component': 'AgentAPI'})
        self.env.process(self._battery_replacement_process(machine_id))
        return True
    def _battery_replacement_process(self, machine_id):
        yield self.env.timeout(BATTERY_REPLACEMENT_TIME)
        logging.info(f"Corrective action: Battery for {machine_id} has been replaced.", extra={'component': 'System'})
        if machine_id in self.machines: self.machines[machine_id].battery_level = 100.0
    def reboot_machine_process(self, machine_id):
        if machine_id not in self.machines: return False
        m = self.machines[machine_id]
        # A stuck process is not 'alive' but we interrupt to be safe
        logging.info(f"AGENT ACTION: Rebooting production process for {machine_id}", extra={'component': 'AgentAPI'})
        if m.production_process.is_alive:
            m.production_process.interrupt()
        m.production_process = self.env.process(m.run_production())
        return True

def fault_injector(env, all_components):
    while True:
        yield env.timeout(random.expovariate(1.0 / MEAN_TIME_BETWEEN_FAULTS))
        target_component = random.choice(all_components)
        if isinstance(target_component, Sensor): target_component.induce_fault(random.choice(SENSOR_FAULT_TYPES))
        elif isinstance(target_component, Actuator): target_component.induce_fault(random.choice(ACTUATOR_FAULT_TYPES))

def setup_simulation_from_csv(env, machines_csv, schedules_csv):
    machines_df = pd.read_csv(machines_csv); schedules_df = pd.read_csv(schedules_csv)
    schedules = (schedules_df.groupby('schedule_id')[['product_id', 'process_time']].apply(lambda g: list(g.itertuples(index=False, name=None))).to_dict())
    network = Network(env); machines = {}; all_sensors = []; all_actuators = []
    for _, row in machines_df.iterrows():
        machine_id = row['machine_id']; schedule_id = row['schedule_id']
        if schedule_id not in schedules: continue
        machine = Machine(env, machine_id, schedules[schedule_id], network)
        machines[machine_id] = machine; all_sensors.extend(machine.sensors); all_actuators.extend(machine.actuators)
        logging.info(f"Created machine: {machine_id}", extra={'component': 'Setup'})
    for _, row in machines_df.iterrows():
        machine_id = row['machine_id']; next_machine_id = row['next_machine_id']
        if pd.notna(next_machine_id) and machine_id in machines and next_machine_id in machines:
            machines[machine_id].next_machine = machines[next_machine_id]
            logging.info(f"Linked {machine_id} -> {next_machine_id}", extra={'component': 'Setup'})
    return machines, all_sensors + all_actuators

def dummy_agent_process(env, api):
    logging.info("Dummy Agent process started.", extra={'component': 'DummyAgent'})
    while True:
        yield env.timeout(15)
        all_machines = api.list_all_machines()
        logging.info(f"Agent running check cycle. Monitoring: {all_machines}", extra={'component': 'DummyAgent'})
        for machine_id in all_machines:
            status = api.get_machine_status(machine_id)
            if not status: continue
            
            # Proactive battery replacement
            if status['battery_level'] < LOW_BATTERY_THRESHOLD + 5:
                logging.info(f"Agent detected low battery on {machine_id}", extra={'component': 'DummyAgent'})
                api.dispatch_battery_replacement(machine_id)

            # Detect and fix stuck actuator
            if status['is_stuck']:
                logging.warning(f"Agent detected {machine_id} is STUCK!", extra={'component': 'DummyAgent'})
                # Step 1: Fix the component that is causing the blockage
                api.recalibrate_component(machine_id, 'actuator', 'PartFeeder')
                # Step 2: Reboot the machine's main process
                api.reboot_machine_process(machine_id)

if __name__ == "__main__":
    print("--- Starting CPS Simulation with Agent API ---")
    print("--- Check 'simulation.log' for detailed output ---")
    
    # (create_config_files() function can be included here if needed)
    
    env = simpy.Environment()
    machines, all_components = setup_simulation_from_csv(env, 'machines.csv', 'schedules.csv')
    
    if machines:
        agent_api = AgentAPI(env, machines)
        env.process(fault_injector(env, all_components))
        env.process(dummy_agent_process(env, agent_api))
        env.run(until=200)
    
    print("\n--- Simulation Finished ---")