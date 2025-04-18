# Vehicle Routing Problem (VRP) using Ant Colony Optimization (ACO)

## Overview
An Ant Colony Optimization (ACO) algorithm is implemented to solve the Vehicle Routing Problem (VRP). In this problem, a fleet of vehicles delivers shipments from a central depot to multiple customer nodes while adhering to vehicle capacity constraints. The depot is represented as a single central node (node type 0), and the customers (node type 1) have associated demands. All vehicles are identical and start and end their routes at the depot. The goal is to generate a set of routes that:
- Deliver all customer demands.
- Minimize the total travel distance.
- Minimize the number of vehicles (routes) when possible.

## Project Structure
- **`vrp_aco.py`**: Contains the implementation of the ACO algorithm for solving the VRP.
- **XML Instance Files**: Example instance files (e.g., `data_32.xml`, `data_72.xml`, and `data_422.xml`) are located in the `hw2/` directory. These files provide input data in XML format.
- **`output/` Directory**: Stores the combined visualization images for each instance. The output files are named based on the input filename (e.g., `data_32.jpg`).

## XML Input File Format
Each XML instance file must follow this structure:
- **Nodes**: The `<network>/nodes` section defines each node by its `id`, `type`, and coordinates `<cx>` and `<cy>`. The depot (node type "0") is stored as the first node with a demand of 0. Customer nodes have their demands updated from the `<requests>` section.
- **Fleet Details**: The `<fleet/vehicle_profile>` section provides the vehicle capacity and identifies the depot from `<departure_node>`.
- **Requests**: The `<requests>` section specifies customer demands. Nodes without a corresponding request (e.g., the depot) are assigned a demand of 0.

## Algorithm Details
The ACO algorithm simulates multiple “ants” that probabilistically build solutions based on:
- **Pheromone Levels**: Indicate the desirability of edges based on previous successful routes.
- **Heuristic Information**: Computed as the inverse of the Euclidean distance between nodes.

### Key Steps
1. **Parsing the Input**  
    The XML file is parsed to extract the nodes, vehicle capacity, and customer demands. The depot is ensured to be the first element with a demand of 0.

2. **Distance Matrix Computation**  
    A 2D Euclidean distance matrix is computed between every pair of nodes. This matrix is used to calculate route costs and guide solution construction.

3. **Solution Construction**  
    For each ant, a solution route is built iteratively:
    - Each route starts at the depot.
    - The next customer is selected based on the current pheromone level (raised to a power, `alpha`) and the heuristic information (raised to a power, `beta`), using a roulette wheel selection.
    - The total demand on a route is ensured not to exceed the vehicle capacity.
    - Each route ends by returning to the depot.

4. **Cost Calculation and Pheromone Update**  
    The total cost is computed by summing the distances of all routes in a solution. After constructing solutions for all ants in an iteration:
    - A fraction of the existing pheromone is evaporated across all edges.
    - New pheromone is deposited inversely proportional to the route cost, reinforcing more effective (shorter) routes.

5. **Convergence and Visualization**  
    The best overall solution cost is tracked across iterations. At the end of the iterations:
    - A map showing the best VRP routes is generated.
    - A convergence graph depicting the best cost per iteration is produced.
    - These plots are combined into one figure and saved in the `output/` folder.

## ACO Parameters
I carefully chose the ACO parameters to balance exploration with convergence:
- **Number of Ants (`num_ants`)**: I set this to **20** to provide sufficient exploration while keeping computation time reasonable.
- **Number of Iterations (`num_iterations`)**: I use **100 iterations**. This number allows the pheromone trails to converge while maintaining acceptable runtime.
- **Pheromone Influence (`alpha`)**: I set `alpha` to **1.0** so that the pheromone trails have a moderate impact on the selection process.
- **Heuristic Influence (`beta`)**: I choose `beta` as **4.0** to emphasize the importance of the inverse distance, thus favoring shorter hops.
- **Evaporation Rate (`evaporation_rate`)**: I use a value of **0.2**. This high rate ensures that outdated routes lose their influence rapidly, promoting new explorations.
- **Random Seed**: I use `random.seed(42)` to ensure that my experiments are reproducible.



## Requirements
- Python 3.x
- Matplotlib (for plotting)
- Standard Python libraries: `xml.etree.ElementTree`, `math`, `random`, `os`, `sys`.

## How to Run the Code
I run the script from the command line by specifying an XML instance file as an argument. For example:

```
pip install -r requirements.txt
python vrp_aco.py hw2/data_32.xml
```

When I run the script, it:
- Parses the XML and builds the necessary data structures.
- Executes the ACO algorithm with the preset parameters.
- Prints the iteration progress and the best solution details.
- Saves the combined output figure (which includes the VRP routes map and the convergence graph) in the `output/` folder.


## Experimental Results
Three example runs have been performed:

### **Data Instance: `data_32.xml`**
- **Parsed Nodes:** 32 (Depot is node 1).
- **Vehicle Capacity:** 100.0.
- **ACO Convergence:**  
  - Initial best cost started at 1155.29 and converged down to 855.09.
- **Solution:**  
  - **Routes:** 5 routes are generated. For example, one route is `[1, 21, 6, 26, 11, 16, 10, 23, 19, 9, 30, 1]`.
- **Output:**  
  - Convergence graph and route visualization saved in `output/data_32.jpg`.

### **Data Instance: `data_72.xml`**
- **Parsed Nodes:** 72 (Depot is node 1).
- **Vehicle Capacity:** 30000.0.
- **ACO Convergence:**  
  - The best cost converged to 302.65 after 100 iterations.
- **Solution:**  
  - **Routes:** 4 routes are generated; one example is `[1, 21, 30, 24, 27, 25, 26, 31, 22, 23, 29, 28, 48, 49, 45, 43, 44, 47, 54, 46, 53, 71, 52, 50, 51, 40, 69, 1]`.
- **Output:**  
  - Convergence graph and route visualization saved in `output/data_72.jpg`.

### **Data Instance: `data_422.xml`**
- **Parsed Nodes:** 421 (Depot is node 421).
- **Vehicle Capacity:** 200.0.
- **ACO Convergence:**  
  - The cost converged to 2294.93 over 100 iterations.
- **Solution:**  
  - **Routes:** 38 routes are generated. The output details a series of routes (e.g., Route 1: `[421, 96, 95, ... ,421]`).
- **Output:**  
  - Convergence graph and route visualization saved in `output/data_422.jpg`.
