import sys
import os
import math
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple

#class for representing the vrp instance from an xml file.
class VRPInstance:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.nodes, self.capacity = self.parse_xml()  #parse xml file and get nodes and vehicle capacity.
        self.demands = [node[3] for node in self.nodes]  #extract demands from nodes.
        self.distances = self.compute_distance_matrix()  #compute distance matrix between nodes.

    #parse the xml file and extract nodes and capacity.
    def parse_xml(self) -> Tuple[List[Tuple[int, float, float, float]], float]:
        try:
            tree = ET.parse(self.filename)  #attempt to parse xml file.
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            sys.exit(1)
        root = tree.getroot()  #get root element.
        nodes_data = {}
        try:
            node_elements = root.find("./network/nodes").findall("node")  #locate node elements in xml.
        except AttributeError:
            print("Error: <network>/<nodes> section not found.")
            sys.exit(1)
        for node in node_elements:
            try:
                node_id = int(node.attrib["id"])  #read node id.
                node_type = node.attrib.get("type", "1")  #get node type (default to 1).
                x = float(node.find("cx").text)  #read x-coordinate.
                y = float(node.find("cy").text)  #read y-coordinate.
            except (KeyError, AttributeError, ValueError) as e:
                print(f"Error processing node: {ET.tostring(node, encoding='unicode')}. Error: {e}")
                continue
            nodes_data[node_id] = {"id": node_id, "x": x, "y": y, "type": int(node_type), "demand": 0.0}
        try:
            vehicle_profile = root.find("./fleet/vehicle_profile")  #find vehicle profile section.
            capacity = float(vehicle_profile.find("capacity").text)  #get vehicle capacity.
            depot_id = int(vehicle_profile.find("departure_node").text)  #get depot id.
        except (AttributeError, ValueError) as e:
            print("Error processing fleet information.")
            sys.exit(1)
        requests_section = root.find("./requests")  #find the requests section.
        if requests_section is not None:
            for request in requests_section.findall("request"):
                try:
                    node_id = int(request.attrib["node"])  #read node id for request.
                    quantity = float(request.find("quantity").text)  #read demand quantity.
                except (KeyError, AttributeError, ValueError) as e:
                    print(f"Error processing request: {ET.tostring(request, encoding='unicode')}. Error: {e}")
                    continue
                if node_id in nodes_data:
                    nodes_data[node_id]["demand"] = quantity  #assign demand to corresponding node.
                else:
                    print(f"Warning: Request for unknown node {node_id}.")
        for nid, node in nodes_data.items():
            if node["demand"] > capacity:
                print(f"Error: Demand for node {nid} ({node['demand']}) exceeds capacity ({capacity}).")
                sys.exit(1)
        depot = [nodes_data[depot_id]]  #set depot as the starting node.
        customers = [node for nid, node in nodes_data.items() if nid != depot_id]  #get all nodes except depot.
        customers.sort(key=lambda n: n["id"])  #sort customers by id.
        nodes = [(nd["id"], nd["x"], nd["y"], nd["demand"]) for nd in depot + customers]  #create final list of nodes.
        return nodes, capacity

    #compute the distance matrix between all nodes.
    def compute_distance_matrix(self) -> List[List[float]]:
        n = len(self.nodes)
        distances = [[0.0 for _ in range(n)] for _ in range(n)]  #initialize n x n distance matrix.
        for i in range(n):
            for j in range(n):
                distances[i][j] = 0.0 if i == j else self.euclidean_distance(self.nodes[i][1], self.nodes[i][2],
                                                                              self.nodes[j][1], self.nodes[j][2])
        return distances

    #calculate euclidean distance between two points.
    def euclidean_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

#class implementing the ant colony optimization algorithm.
class AntColony:
    def __init__(self, distances: List[List[float]], demands: List[float], capacity: float, num_ants: int,
                 num_iterations: int, alpha: float, beta: float, evaporation_rate: float) -> None:
        self.distances = distances
        self.demands = demands
        self.capacity = capacity
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  #influence of pheromone.
        self.beta = beta  #influence of distance.
        self.evaporation_rate = evaporation_rate  #pheromone evaporation rate.
        self.n = len(demands)
        self.pheromone = [[1.0 for _ in range(self.n)] for _ in range(self.n)]  #initialize pheromone levels.
        self.best_solution = None
        self.best_cost = float('inf')
        self.convergence = []  #track best cost over iterations.

    #select next node based on probabilities using roulette wheel selection.
    def roulette_selection(self, candidates: List[int], probabilities: List[float]) -> int:
        total = sum(probabilities)
        if total == 0:
            return random.choice(candidates)  #if probabilities sum to zero, choose randomly.
        r = random.uniform(0, total)
        cumulative = 0.0
        for candidate, prob in zip(candidates, probabilities):
            cumulative += prob
            if cumulative >= r:
                return candidate
        return candidates[-1]

    #construct a solution (set of routes) starting and ending at the depot.
    def construct_solution(self, depot_index: int = 0) -> List[List[int]]:
        solution = []
        unvisited = set(range(1, self.n))  #all nodes except depot.
        while unvisited:
            route = [depot_index]  #start route from depot.
            remaining_capacity = self.capacity
            current = depot_index
            while True:
                feasible = [j for j in unvisited if self.demands[j] <= remaining_capacity]  #select feasible nodes.
                if not feasible:
                    break  #no feasible node available.
                probs = []
                for j in feasible:
                    tau = self.pheromone[current][j] ** self.alpha
                    eta = (1.0 / self.distances[current][j]) ** self.beta if self.distances[current][j] > 0 else 0.0
                    probs.append(tau * eta)
                next_customer = self.roulette_selection(feasible, probs)  #select next node based on computed probabilities.
                route.append(next_customer)
                unvisited.remove(next_customer)  #mark node as visited.
                remaining_capacity -= self.demands[next_customer]
                current = next_customer
            route.append(depot_index)  #return to depot at end of route.
            solution.append(route)
        return solution

    #compute total travel cost for a given solution.
    def compute_cost(self, solution: List[List[int]]) -> float:
        total_cost = 0.0
        for route in solution:
            for i in range(len(route) - 1):
                total_cost += self.distances[route[i]][route[i+1]]
        return total_cost

    #update pheromone levels based on the solutions found in the current iteration.
    def update_pheromone(self, solutions: List[List[List[int]]], costs: List[float]) -> None:
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.evaporation_rate)  #apply pheromone evaporation.
        for sol, cost in zip(solutions, costs):
            if cost == 0:
                continue  #skip pheromone update if cost is zero.
            deposit = 1.0 / cost
            for route in sol:
                for k in range(len(route) - 1):
                    i = route[k]
                    j = route[k+1]
                    self.pheromone[i][j] += deposit  #deposit pheromone on the path.
                    self.pheromone[j][i] += deposit

    #run the ant colony optimization algorithm.
    def run(self) -> Tuple[List[List[int]], float, List[float]]:
        for it in range(self.num_iterations):
            solutions = []
            costs = []
            for ant in range(self.num_ants):
                sol = self.construct_solution()  #generate solution for each ant.
                cost = self.compute_cost(sol)  #calculate cost of the solution.
                solutions.append(sol)
                costs.append(cost)
                if cost < self.best_cost:
                    self.best_cost = cost  #update best cost found.
                    self.best_solution = sol  #update best solution.
            self.convergence.append(self.best_cost)  #record the best cost for current iteration.
            self.update_pheromone(solutions, costs)  #update pheromone based on solutions.
            print(f"Iteration {it+1}/{self.num_iterations} - Best Cost: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost, self.convergence

#plot best solution routes and convergence curve.
def plot_results(best_solution: List[List[int]], convergence: List[float],
                 nodes: List[Tuple[int, float, float, float]], output_filename: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    for route in best_solution:
        xs = [nodes[i][1] for i in route]
        ys = [nodes[i][2] for i in route]
        ax1.plot(xs, ys, marker='o', linestyle='-')  #plot the route.
        for i in route:
            ax1.text(nodes[i][1], nodes[i][2], str(nodes[i][0]), fontsize=9, color='red')  #annotate node id.
    ax1.set_title("Best VRP Routes")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True)
    ax2.plot(convergence, marker='o', linestyle='-')  #plot the convergence curve.
    ax2.set_title("ACO Convergence")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Cost")
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Figure saved to {output_filename}")

#get command line arguments.
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve VRP using Ant Colony Optimization")
    parser.add_argument("instance_file", help="XML file containing the VRP instance")
    parser.add_argument("--ants", type=int, default=10, help="Number of ants (default: 10)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations (default: 100)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Pheromone influence (default: 1.0)")
    parser.add_argument("--beta", type=float, default=2.0, help="Distance influence (default: 2.0)")
    parser.add_argument("--evaporation_rate", type=float, default=0.5, help="Evaporation rate (default: 0.5)")
    return parser.parse_args()  #return parsed arguments.

#main function to run the optimization process.
def main() -> None:
    args = get_args()matplotlib
argparse
    instance_file = args.instance_file
    if not os.path.exists(instance_file):
        print(f"Error: File {instance_file} does not exist.")
        sys.exit(1)
    random.seed(42)  #set seed for reproducibility.
    vrp_instance = VRPInstance(instance_file)  #create vrp instance from xml file.
    print(f"Parsed {len(vrp_instance.nodes)} nodes. Depot is node {vrp_instance.nodes[0][0]}.")
    print(f"Vehicle capacity: {vrp_instance.capacity}")
    aco = AntColony(vrp_instance.distances, vrp_instance.demands, vrp_instance.capacity,
                    args.ants, args.iterations, args.alpha, args.beta, args.evaporation_rate)
    print("Starting ACO algorithm...")
    best_solution, best_cost, convergence = aco.run()  #execute the ant colony optimization algorithm.
    print("\nBest overall solution:")
    for idx, route in enumerate(best_solution):
        route_ids = [vrp_instance.nodes[i][0] for i in route]
        print(f"  Route {idx+1}: {route_ids}")
    print(f"Total cost (distance): {best_cost:.2f}")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)  #create output directory if not exists.
    basename = os.path.splitext(os.path.basename(instance_file))[0]
    output_filename = os.path.join(output_dir, f"{basename}.jpg")
    plot_results(best_solution, convergence, vrp_instance.nodes, output_filename)  #plot and save the results.

if __name__ == "__main__":
    main()
