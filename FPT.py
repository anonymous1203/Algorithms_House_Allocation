import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Problem parameters
n_agents = 5
houses_list = [13]
m_houses=houses_list[0]
num_instances = 100

def instance(num_agents, num_houses, max_value=100):
    M = np.zeros((num_agents, num_houses), dtype=int)
    for agent in range(num_agents):
        k = np.random.randint(4, 6)  # Between 0 and 5 inclusive
        if k > 0:
            liked_houses = np.random.choice(num_houses, size=k, replace=False)
            M[agent, liked_houses] = np.random.randint(1, max_value + 1, size=k)
    return M

valuations = instance(n_agents, m_houses)




# Example cardinal valuations: rows=agents, columns=houses
# cardinal_valuations = np.array([
#     [10, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # Agent 0
#     [9, 10, 6, 5, 4, 3, 2, 1, 0, 7],  # Agent 1
#     [8, 10, 1, 9, 3, 2, 1, 0, 6, 5],  # Agent 2
#     [7, 10, 9, 1, 2, 1, 0, 5, 4, 3],  # Agent 3
#     [6, 10, 4, 3, 1, 9, 8, 7, 0, 1],  # Agent 4
# ])

# cardinal_valuations = np.array([[10, 6, 6, 2, 0,0,0,0,0,0], [3,6,5,3, 0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]])


# Create Gurobi model
model = gp.Model("min_envy_cardinal")

# Decision variables: x[i,j] = 1 if agent i assigned house j
x = model.addVars(n_agents, m_houses, vtype=GRB.BINARY, name="x")

# e[a] = 1 if agent a is envious of at least one other agent
e = model.addVars(n_agents, vtype=GRB.BINARY, name="e")

# y[a,b] = 1 if agent a envies agent b
y = model.addVars(n_agents, n_agents, vtype=GRB.BINARY, name="y")

# Each agent gets exactly one house
for i in range(n_agents):
    model.addConstr(gp.quicksum(x[i, j] for j in range(m_houses)) == 1)

# Each house assigned to at most one agent
for j in range(m_houses):
    model.addConstr(gp.quicksum(x[i, j] for i in range(n_agents)) <= 1)

# Define envy constraints
for a in range(n_agents):
    for b in range(n_agents):
        if a == b:
            model.addConstr(y[a, b] == 0)
            continue
        for j in range(m_houses):
            for k in range(m_houses):
                if valuations[a, j] > valuations[a, k]:
                    model.addConstr(y[a, b] >= x[b, j] + x[a, k] - 1)

# Link e[a] with y[a,b]: e[a] >= y[a,b] for all b != a
for a in range(n_agents):
    for b in range(n_agents):
        if a != b:
            model.addConstr(e[a] >= y[a, b])

# Stage 1: Minimize sum of envious agents
model.setObjective(gp.quicksum(e[a] for a in range(n_agents)), GRB.MINIMIZE)
model.optimize()

if model.status == GRB.OPTIMAL:
    E_star = int(round(model.objVal))
    print(f"Minimum number of envious agents: {E_star}")

    # Stage 2: Maximize welfare, subject to minimum envy
    model.addConstr(gp.quicksum(e[a] for a in range(n_agents)) <= E_star)
    # Set new objective: maximize total welfare
    model.setObjective(
        gp.quicksum(valuations[i, j] * x[i, j] for i in range(n_agents) for j in range(m_houses)),
        GRB.MAXIMIZE
    )
    model.optimize()

    # Output results
if model.status == GRB.OPTIMAL:
    print(f"\nWelfare-maximizing allocation with {E_star} envious agents:")
    
    # Collect allocation as list of tuples
    optimal_allocation = []
    for i in range(n_agents):
        for j in range(m_houses):
            if x[i, j].X > 0.5:
                optimal_allocation.append((i, j))  # (agent, house)
    
  
    print("\nEnvious agents:")
    for a in range(n_agents):
        print(f" Agent {a}: {'Envious' if e[a].X > 0.5 else 'Not envious'}")
    print(f"Total welfare: {int(round(model.ObjVal))}")
else:
    print("No optimal solution found.")


def max_weighted_matching(M):
    G = nx.Graph()
    for agent in range(M.shape[0]):
        for house in range(M.shape[1]):
            G.add_edge(agent, n_agents + house, weight=M[agent][house])
    matching = nx.max_weight_matching(G, maxcardinality=True)
    allocation = []
    for a, h in matching:
        if a < n_agents:
            agent, house = a, h - n_agents
        else:
            agent, house = h, a - n_agents
        allocation.append((agent, house))
    allocation.sort()
    return allocation

max_welfare_allocation = max_weighted_matching(valuations)

print(valuations)

print(f"Optimal Allocation: {optimal_allocation}")
print("Maximum Welfare Allocation:", max_welfare_allocation)



def compute_symmetric_difference(A_dict, P):
    def is_house(node): return node.startswith('h')
    def is_agent(node): return node.startswith('i')

    edges = []
    for i in range(len(P)-1):
        u, v = P[i], P[i+1]
        if is_house(u) and is_agent(v):
            edges.append((v, u)) 
        elif is_agent(u) and is_house(v):
            edges.append((u, v))
    new_A = A_dict.copy()
    for agent, house in edges:
        if new_A.get(agent) == house:
            del new_A[agent]
        else:
            if agent in new_A:
                del new_A[agent]
            new_A[agent] = house
    return new_A


def calculate_envy(allocation_dict, valuations):
    if isinstance(allocation_dict, list):
        allocation_dict = {f'i{a}': f'h{h}' for (a, h) in allocation_dict}
    else:
        allocation_dict = allocation_dict

    allocation = [(int(a[1:]), int(h[1:])) for a, h in allocation_dict.items()]
    envy_count = 0
    for a in range(valuations.shape[0]):
        agent_houses = [h for (agent, h) in allocation if agent == a]
        if not agent_houses:
            continue
        current_house = agent_houses[0]
        current_value = valuations[a, current_house]
        
        for (other_agent, other_house) in allocation:
            if valuations[a, other_house] > current_value:
                envy_count += 1
                break
    return envy_count

G = nx.Graph()
n_agents, m_houses = valuations.shape
for agent in range(n_agents):
    for house in range(m_houses):
        if valuations[agent, house] > 0:
            G.add_edge(f'i{agent}', f'h{house}')


def find_alternating_components(A, A_hat, valuations):
    A_edges = {(f'i{a}', f'h{h}') for (a, h) in A}
    A_hat_edges = {(f'i{a}', f'h{h}') for (a, h) in A_hat}
    sym_diff = A_edges.symmetric_difference(A_hat_edges)

    # Induce subgraph on symmetric difference edges
    H = G.edge_subgraph(sym_diff)
    
    components = []
    for component in nx.connected_components(H):
        subgraph = H.subgraph(component)
        
        try:
            cycle = nx.find_cycle(subgraph)
            nodes = [cycle[0][0], cycle[0][1]]
            for edge in cycle[1:]:
                nodes.append(edge[1])
            components.append(nodes)
        except nx.NetworkXNoCycle:
            endpoints = [n for n, d in subgraph.degree() if d == 1]
            if len(endpoints) == 2:
                path = nx.shortest_path(subgraph, endpoints[0], endpoints[1])
                components.append(path)
    return components



def apply_all_components(initial_allocation, paths):
    A_dict = {f'i{a}': f'h{h}' for (a, h) in initial_allocation}
    for path in paths:
        A_dict = compute_symmetric_difference(A_dict, path)
    return [(int(a[1:]), int(h[1:])) for a, h in A_dict.items()]

alternating_components = find_alternating_components(optimal_allocation, max_welfare_allocation, valuations)
A_hat_paths = apply_all_components(max_welfare_allocation, alternating_components)



def good_coloring(G, A, A_hat, n_agents, m_houses):
    A_edges = {(a, n_agents + h) for a, h in A}
    A_hat_edges = {(a, n_agents + h) for a, h in A_hat}
    T = A_edges.symmetric_difference(A_hat_edges)
    S = set()
    for u, v in T:
        S.add(u)
        S.add(v)
    vertex_colors = {node: 'blue' for node in G.nodes()}  # Default to blue
    edge_colors = {}
    for node in S:
        vertex_colors[node] = 'red'
    for u, v in G.edges():
        edge = (u, v)
        if edge in T or (v, u) in T:
            edge_colors[edge] = 'red'
        elif u in S and v in S:
            edge_colors[edge] = 'green'
        elif (u in S) ^ (v in S):  # XOR: exactly one endpoint in S
            edge_colors[edge] = 'blue'
        else:
            edge_colors[edge] = 'gray'  # Default for edges not involving S
    return vertex_colors, edge_colors



# def find_feasible_components(G, vertex_colors, edge_colors, A_hat, valuations):
#     red_edges = [e for e, color in edge_colors.items() if color == 'red']
#     G_red = G.edge_subgraph(red_edges)
#     components = list(nx.connected_components(G_red))
#     # print(list(nx.connected_components(G_red)))
#     paths_in_red = []
#     for component in components:
#         induced_subgraphs = G.subgraph(component) 
#         paths_in_red.append(list(induced_subgraphs.edges))
#     # print(paths_in_red)
#     # print("red components in G look like this:", components)
#     feasible = []
    
#     for comp in components:
#         # Condition 1: No green/blue vertices
#         if any(vertex_colors[node] != 'red' for node in comp):
#             continue
            
#         # Condition 3: No blue edges within component
#         if has_internal_blue_edges(comp, edge_colors):
#             continue
            
#         # Condition 4: Check niceness (simplified check)
#         if not is_nice_component(comp, components, A_hat, valuations):
#             continue
            
#         feasible.append(comp)
    
#     return feasible





def find_feasible_components(G, vertex_colors, edge_colors, A_hat, valuations):
    red_edges = [e for e, color in edge_colors.items() if color == 'red']
    G_red = G.edge_subgraph(red_edges)
    components = list(nx.connected_components(G_red))
    paths_in_red = []
    for component in components:
        induced_subgraphs = G.subgraph(component) 
        paths_in_red.append(list(induced_subgraphs.edges))
    feasible = []
    feasible_paths =[]
    
    for path in paths_in_red:  # Changed iteration target
        # Extract nodes from path edges
        nodes_in_path = {u for edge in path for u in edge}
        
        # Condition 1: All vertices red
        if any(vertex_colors[node] != 'red' for node in nodes_in_path):
            continue
            
        # Condition 3: No internal blue edges
        if has_internal_blue_edges(nodes_in_path, edge_colors):
            continue
            
        # Condition 4: Niceness check
        if not is_nice_component(nodes_in_path, components, A_hat, valuations):
            continue
            
        feasible.append(nodes_in_path)
        feasible_paths.append(path)

    return feasible, feasible_paths



def convert_feasible_paths(feasible_paths, n_agents):
    labeled_paths = []
    for path in feasible_paths:
        # Extract unique nodes from edges
        nodes = {u for edge in path for u in edge}
        
        # Build node sequence (simple linear path for demonstration)
        node_sequence = []
        for edge in path:
            u, v = edge
            if u not in node_sequence:
                node_sequence.append(u)
            if v not in node_sequence:
                node_sequence.append(v)
        
        # Convert to labels
        labeled_nodes = []
        for node in node_sequence:
            if node < n_agents:
                labeled_nodes.append(f'i{node}')
            else:
                labeled_nodes.append(f'h{node - n_agents}')
                
        labeled_paths.append(labeled_nodes)
    
    return labeled_paths


def reorder_labeled_paths(labeled_paths, A_hat):
    # Convert A_hat to dictionary: {agent: house}
    a_hat_mapping = {f'i{a}': f'h{h}' for (a, h) in A_hat}
    
    reordered_paths = []
    for path in labeled_paths:
        # Extract agent and houses
        agent = next(node for node in path if node.startswith('i'))
        houses = [node for node in path if node.startswith('h')]
        
        # Identify A_hat house and other houses
        a_hat_house = a_hat_mapping[agent]
        other_houses = [h for h in houses if h != a_hat_house]
        
        # Create new path: [non-A_hat house, agent, A_hat house]
        if other_houses:  # Ensure there's at least one other house
            new_path = [other_houses[0], agent, a_hat_house]
            reordered_paths.append(new_path)
    
    return reordered_paths




def has_internal_blue_edges(component, edge_colors):
    """Check for blue edges between component vertices."""
    for u in component:
        for v in component:
            if (u, v) in edge_colors and edge_colors[(u, v)] == 'blue':
                return True
    return False

def is_nice_component(current_comp, all_comps, A_hat, valuations):
    # Simplified check - full implementation requires comparing all combinations
    return True

def calculate_envy_reduction(path, A_hat, valuations):
    temp_alloc = apply_all_components(A_hat, path)
    agent_houses = {a: h for a, h in A_hat}
    original_envy = calculate_envy(A_hat, valuations)
    new_envy = calculate_envy(temp_alloc, valuations)
    return original_envy - new_envy

def compute_min_envy(A, A_hat, valuations):
    n_agents = len(A)
    m_houses = valuations.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_agents + m_houses))
    for a in range(n_agents):
        for h in range(m_houses):
            G.add_edge(a, n_agents + h)

    vertex_colors, edge_colors = good_coloring(G, A, A_hat, n_agents, m_houses)
    
    feasible_comps, feasible_paths = find_feasible_components(G, vertex_colors, edge_colors, A_hat, valuations)

    labeled_paths = convert_feasible_paths(feasible_paths, n_agents)
    reordered_labeled_paths = reorder_labeled_paths(labeled_paths, max_welfare_allocation)
    A_algo_output = apply_all_components(max_welfare_allocation, reordered_labeled_paths)
    print("reorderedpaths:", reordered_labeled_paths)
    print("Transformed algo allocation:",A_algo_output)

    
    # 4. Prepare knapsack inputs
    items_n_C = []
    items_r_C = []
    items = []

    r_C = calculate_envy_reduction(reordered_labeled_paths, A_hat, valuations)
        
    for comp in feasible_comps:
        agents = {n for n in comp if n < n_agents}
        n_C = len(agents)
        items_n_C.append(n_C)
        items_r_C.append(r_C)

    
    

    

    for item in range(len(items_n_C)):
        items.append((items_r_C[item], items_n_C[item]))

        
    # 5. Solve knapsack problem

    print(items)
    q = sum(n for _, n in items)

    k = sum(r_c for r_c, _ in items)

    # dp = [0]*(q+1)
    # print(dp)
    # for r, n in items:
    #     for w in range(q, n-1, -1):
    #         if dp[w - n] + r > dp[w]:
    #             dp[w] = dp[w - n] + r


    
    return k, q


k, q = compute_min_envy(optimal_allocation, max_welfare_allocation, valuations)
print("k, q:", k, q)


print("Alternating components:", alternating_components)
print("Transformed allocation:", A_hat_paths)


# def is_nice_subset(S, all_components, original_allocation, valuations):
#     from itertools import chain, combinations
#     original_envy = calculate_envy(original_allocation, valuations)
    
#     # Compute envy after applying S (k1)
#     alloc_after_S = original_allocation.copy()
#     for comp in S:
#         alloc_after_S = compute_symmetric_difference(alloc_after_S, comp)
#     k1 = original_envy - calculate_envy(alloc_after_S, valuations)
    
#     # Get remaining components (T \ S)
#     remaining = [c for c in all_components if c not in S]
    
#     # Generate all subsets of remaining components (including empty)
#     for r in range(len(remaining) + 1):
#         for subset in combinations(remaining, r):
#             # Compute envy after applying subset (k2)
#             alloc_after_subset = original_allocation.copy()
#             for comp in subset:
#                 alloc_after_subset = compute_symmetric_difference(alloc_after_subset, comp)
#             k2 = original_envy - calculate_envy(alloc_after_subset, valuations)
            
#             # Compute combined effect (S ∪ subset)
#             combined = list(S) + list(subset)
#             alloc_combined = original_allocation.copy()
#             for comp in combined:
#                 alloc_combined = compute_symmetric_difference(alloc_combined, comp)
#             k_combined = original_envy - calculate_envy(alloc_combined, valuations)
            
#             # Check additive property
#             if k_combined != k1 + k2:
#                 return False
#     return True


