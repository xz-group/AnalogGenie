import pandas as pd
import numpy as np
import os
import sys
# Check current recursion limit
print("Current recursion limit:", sys.getrecursionlimit())

# Increase recursion limit
new_limit = 9000  # Set to a higher value
sys.setrecursionlimit(new_limit)

print("New recursion limit:", sys.getrecursionlimit())

def read_connection_matrix(filename):
    """
    Reads the connection matrix from a CSV file and returns it as a DataFrame.
    """
    return pd.read_csv(filename, index_col=0)

def get_edges_from_path(traversal_path):
    """Extracts the directed edges from a given traversal path."""
    return [(traversal_path[i], traversal_path[i + 1]) for i in range(len(traversal_path) - 1)]

def check_if_path_covers_all_edges_exactly_once(matrix, traversal_path):
    """Checks if the given traversal path covers all directed edges in the connection matrix exactly once."""
    # Extract all edges from the traversal path
    path_edges = get_edges_from_path(traversal_path)

    # Extract all edges from the connection matrix
    graph_edges = set()
    for i in matrix.index:
        for j in matrix.columns:
            if matrix.loc[i, j] == 1:
                graph_edges.add((i, j))

    # Check if all graph edges are covered by path edges exactly once
    if len(path_edges) != len(graph_edges):
        return False

    path_edges_count = {edge: path_edges.count(edge) for edge in path_edges}
    if any(count > 1 for count in path_edges_count.values()):
        return False

    return set(path_edges) == graph_edges  

def all_eulerian_circuits_directed(matrix, start_node='VSS', max_solutions=None):
    """
    Returns all Eulerian circuits in a directed graph derived from an undirected graph defined 
    by a connection matrix. In the constructed directed graph, every directed edge is used 
    exactly once and the circuit starts and ends at start_node.
    
    For each undirected edge (u,v) with u != v, we add two directed edges: u -> v and v -> u.
    For a self-loop (u,u), we add a single directed edge.
    
    Parameters:
        matrix (pandas.DataFrame): A symmetric DataFrame with nodes as both index and columns.
        start_node (str): The starting node for the Eulerian circuit (default is 'VSS').
        max_solutions (int or None): Maximum number of solutions to find. If None, finds all solutions.
        
    Returns:
        List[List[str]]: A list of Eulerian circuits. Each circuit is a list of nodes starting 
                         and ending at start_node, with every directed edge used exactly once.
    """
    # Build the directed graph from the undirected connection matrix.
    # Each node maps to a list of its outgoing neighbors.
    graph = {node: [] for node in matrix.index}
    total_edges = 0  # Total number of directed edges to use in the circuit
    nodes = list(matrix.index)
    n = len(nodes)
    
    # Process self-loops and edges (using upper triangle to avoid double counting)
    for i in range(n):
        node = nodes[i]
        # Self-loop: if present, add one directed edge from node to itself.
        if matrix.at[node, node] == 1:
            graph[node].append(node)
            total_edges += 1
        for j in range(i + 1, n):
            neighbor = nodes[j]
            if matrix.at[node, neighbor] == 1:
                # For an undirected edge, add both directed edges.
                graph[node].append(neighbor)
                graph[neighbor].append(node)
                total_edges += 2

    circuits = []
    
    def backtrack(current, path, used_edges):
        # If we have reached the maximum number of solutions, stop exploring.
        if max_solutions is not None and len(circuits) >= max_solutions:
            return
        
        # If all directed edges have been used and we are back at the start, record the circuit.
        if used_edges == total_edges and current == start_node:
            circuits.append(path.copy())
            return
        
        # Explore each outgoing edge from the current node.
        for neighbor in list(graph[current]):
            # Remove the directed edge from current to neighbor.
            graph[current].remove(neighbor)
            path.append(neighbor)
            
            backtrack(neighbor, path, used_edges + 1)
            
            # Backtrack: restore the edge and remove the last node from the path.
            path.pop()
            graph[current].append(neighbor)
    
    # Start the Eulerian circuit search from start_node.
    backtrack(start_node, [start_node], 0)
    return circuits


# Dictionary of directories and their respective end values
base_dirs = {
    "Dataset": 3502
}

# Loop through each base directory
for base_dir, end in base_dirs.items():
    for i in range(1, end+1):  # Iterate through numbered subdirectories
        max_solutions = 20 if i > 1280 else 200  # Set max_solutions based on directory
        number = str(i)
        dir_path = f"{base_dir}/{number}"
        if not os.path.isdir(dir_path):
            # If it doesn't exist, skip the rest of this loop iteration
            # print(f"Directory '{dir_path}' does not exist. Skipping...")
            continue
        
        connect_dir = os.path.join(base_dir, number, f'Graph{number}.csv')

        # Check if the connection matrix file exists
        if not os.path.exists(connect_dir):
            print(f"Connection matrix file not found: {connect_dir}")
            continue

        try:
            # Read the connection matrix
            connection_matrix = read_connection_matrix(connect_dir)

            all_traversal_paths = all_eulerian_circuits_directed(connection_matrix, start_node='VSS', max_solutions=max_solutions)

            # Check if all paths cover all edges exactly once
            if not all(check_if_path_covers_all_edges_exactly_once(connection_matrix, path) for path in all_traversal_paths):
                print(f"DATA {base_dir}/{number}: Paths do not cover edges exactly once")
                break

            # Pad paths to a fixed length of 1025
            padded_paths = [
                path + ['TRUNCATE'] * (1025 - len(path)) for path in all_traversal_paths if len(path) <= 1025
            ]

            # Output the number of complete paths
            print(f"DATA {base_dir}/{number}: Number of complete paths: {len(padded_paths)}")

            # Check if all paths are unique
            if len(padded_paths) != len(set(tuple(path) for path in padded_paths)):
                print(f"DATA {base_dir}/{number}: Paths are not unique")
                break

            # Save the padded paths to a file
            save_dir = os.path.join(base_dir, number, f'Sequence_total{number}.npy')
            np.save(save_dir, padded_paths)
            print(padded_paths)

        except Exception as e:
            print(f"Error processing {base_dir}/{number}: {e}")