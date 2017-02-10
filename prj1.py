"""

This module is the main module of the A.I. Project #1 - Modified 15-Puzzle

Author: Erbil Oner - 2017
"""


import heapq
import numpy as np
import copy
import argparse
import math
from functools import total_ordering


@total_ordering
class Node:
    def __init__(self, parent, state, action, pathcost, f=0):
        self.parent = parent
        self.state = state
        self.action = action
        self.path_cost = pathcost
        self.f = f
        self.state_text = get_state_text(state)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.f == other.f:
                return self.path_cost == other.path_cost
            return self.f == other.f
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other,self.__class__):
            if self.f == other.f:
                return self.path_cost < other.path_cost
            return self.f < other.f
        return NotImplemented


def main():
    """
    This is the main function.
    This function calls 'tester' function of module 'puzzletester'
    puzzletester.tester() function includes all interactions and analysis part.
    """
    arg = get_args()
    data = get_data(arg['file'][0]) # read initial state from file

    solve_puzzle(data, arg)


def get_args():
    parser = argparse.ArgumentParser(description='Modified 15-puzzle solver')

    parser.add_argument('--search', '-s', nargs=1, choices=["ils", "ucs", "astar"], required=True,
                        help="Search method\n"
                             "<ils> for Iterative Lengthening Search\n"
                             "<ucs> for Uniform-Cost Search\n"
                             "<astar> for A* Search")

    parser.add_argument('--file', '-f', nargs=1, type=str, required=True, help="Input file name")
    parser.add_argument('--eval', '-e', nargs=1, type=int, choices=[1, 2], help="Heuristic Type\n"
                                                                               "<1> Misplaced Tiles\n"
                                                                               "<2> My heuristic")

    arg = parser.parse_args()
    return vars(arg)


def get_data(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, dtype=int)
    return data.tolist()


def solve_puzzle(data, arg):
    node_initial = Node(None, data, None, 0)

    print("Initial state for (a):")
    print(np.matrix(node_initial.state))
    print("---------------")

    # Error check
    if arg['search'][0] == "astar" and not arg['eval']:
        raise argparse.ArgumentError(message="<astar> cannot be used without specifying the evaluation function")

    # Search methods builder
    if arg['search'][0] == "ucs":
        print("Uniform Cost Search:")
        result, num_of_exp_nodes, msn = ucs(node_initial, True)
    elif arg['search'][0] == "ils":
        result, num_of_exp_nodes, msn = ils(node_initial, math.inf, True)
    elif arg['search'][0] == "astar":
        h_dict = {1: "h1", 2: "h2"}
        heur_type = arg['eval'][0]
        result, num_of_exp_nodes, msn = astar(node_initial, h_dict[heur_type], True)

    depth = report_path(result)
    print("Number of expanded nodes: " + str(num_of_exp_nodes))
    print("Solution's depth:" + str(depth))
    print("Maximum # of nodes stored in memory: " + str(msn))
    eff_bf = calculate_bfactor(depth, num_of_exp_nodes)
    print("Effective branching factor: " + str(eff_bf))


def get_state_text(state):
    """Return the string representation of a given state
    Example:
    1  2  3  5
    12 6  8  7
    0  13 14 9
    10 11 15 4

    to

    '1-2-3-5-12-6-8-7-0-13-14-9-10-11-15-4'

    :param state: 2D list
    :return: string
    """
    text = ''
    for i in range(np.size(state, 0)):
        for j in range(np.size(state, 0)):
            text = text + str(state[i][j]) + '-'

    text = text[:-1]
    return text


def get_actions(state):
    """ Return actions for given state.

    Argument:
        state -- the given state of an agent
    """

    state_np = np.asarray(state)

    row, col = np.where(state_np == 0)

    row = row[0]
    col = col[0]

    if row == 0 and col == 0:
        return ["goDownRight", "goRight", "goDown"]
    elif row == 0 and col == 3:
        return ["goDownLeft", "goLeft", "goDown"]
    elif row == 3 and col == 0:
        return ["goUpRight", "goRight", "goUp"]
    elif row == 3 and col == 3:
        return ["goUpLeft", "goLeft", "goUp"]
    elif row == 0:
        return ["goDownLeft", "goDownRight", "goLeft", "goRight", "goDown"]
    elif row == 3:
        return ["goUpLeft", "goUpRight", "goLeft", "goRight", "goUp"]
    elif col == 0:
        return ["goUpRight", "goDownRight", "goRight", "goUp", "goDown"]
    elif col == 3:
        return ["goDownLeft", "goUpLeft", "goLeft", "goUp", "goDown"]
    else:
        return ["goUpLeft", "goUpRight", "goDownLeft", "goDownRight", "goLeft", "goRight", "goUp", "goDown"]


def result_cost(action):
    """ Return the cost of a given action

    Arguments:
         action -- (String) A movement of an agent
    """
    if action == "goUpLeft" or action == "goUpRight" or action == "goDownLeft" or action == "goDownRight":
        return 1.5
    else:
        return 1


def result_state(state, action):
    """ Return a state for given action

    Arguments:
        state -- Current state of an agent
        action -- An action that applied to the state

    :param state: list 2D
    :param action: string
    :return: list 2D
    """
    state_np = np.asarray(state)

    row, col = np.where(state_np == 0)

    row = row[0]
    col = col[0]

    new_state = copy.deepcopy(state)

    if action == "goLeft":
        target = new_state[row][col - 1]
        new_state[row][col] = target
        new_state[row][col - 1] = 0
    elif action == "goRight":
        target = new_state[row][col + 1]
        new_state[row][col] = target
        new_state[row][col + 1] = 0
    elif action == "goUp":
        target = new_state[row - 1][col]
        new_state[row][col] = target
        new_state[row - 1][col] = 0
    elif action == "goDown":
        target = new_state[row + 1][col]
        new_state[row][col] = target
        new_state[row + 1][col] = 0
    elif action == "goUpLeft":
        target = new_state[row - 1][col - 1]
        new_state[row][col] = target
        new_state[row - 1][col - 1] = 0
    elif action == "goUpRight":
        target = new_state[row - 1][col + 1]
        new_state[row][col] = target
        new_state[row - 1][col + 1] = 0
    elif action == "goDownLeft":
        target = new_state[row + 1][col - 1]
        new_state[row][col] = target
        new_state[row + 1][col - 1] = 0
    elif action == "goDownRight":
        target = new_state[row + 1][col + 1]
        new_state[row][col] = target
        new_state[row + 1][col + 1] = 0

    return new_state


def get_child(node, action, h=None):
    """Return a child node for given parent

    Arguments:
        node -- Parent node
        action -- An action that applied to the state
        h -- heuristic type for A* algorithm

    :param node: Node
    :param action: string
    :param h: string
    :return: Node
    """

    child_state = result_state(node.state, action)
    child_parent = node
    child_action = action

    if h == "h1":
        child_path_cost = node.path_cost + result_cost(action)
        f = child_path_cost + get_h1(child_state)
    elif h == "h2":
        child_path_cost = node.path_cost + result_cost(action)
        f = child_path_cost + get_h2(child_state)
    else:
        child_path_cost = node.path_cost + result_cost(action)
        f = child_path_cost

    return Node(child_parent, child_state, child_action, child_path_cost, f)


def goal_test(state):
    """ Test the given state

    :param state: list 2D
    :return: bool
    """

    goal_tuple = ((1, 2, 3, 4),
                  (12, 13, 14, 5),
                  (11, 0, 15, 6),
                  (10, 9, 8, 7))

    goal_hash = hash(goal_tuple)

    test_tuple = tuple([tuple(row) for row in state])
    test_hash = hash(test_tuple)

    return True if test_hash == goal_hash else False


def get_h1(state):
    """ Return the sum of misplaced tiles for given state

    :param state: list 2D
    :return: int
    """
    goal_list = [[1, 2, 3, 4],
                 [12, 13, 14, 5],
                 [11, 0, 15, 6],
                 [10, 9, 8, 7]]

    state_np = np.asarray(state)
    goal_np = np.asarray(goal_list)

    misplaced_tiles = np.sum(state_np != goal_np)

    if state[2][1] != 0:
        misplaced_tiles -= 1

    return misplaced_tiles


def get_h2(state):
    """Return the result of my heuristic function

    :param state: list 2D
    :return: int
    """

    total_dist = 0
    for row_idx, row in enumerate(state):
        for col_idx, col in enumerate(row):
            if col == 0:
                continue
            elif col == 1:
                row_diff, col_diff = abs(row_idx - 0), abs(col_idx - 0)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 2:
                row_diff, col_diff = abs(row_idx - 0), abs(col_idx - 1)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 3:
                row_diff, col_diff = abs(row_idx - 0), abs(col_idx - 2)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 4:
                row_diff, col_diff = abs(row_idx - 0), abs(col_idx - 3)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 5:
                row_diff, col_diff = abs(row_idx - 1), abs(col_idx - 3)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 6:
                row_diff, col_diff = abs(row_idx - 2), abs(col_idx - 3)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 7:
                row_diff, col_diff = abs(row_idx - 3), abs(col_idx - 3)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 8:
                row_diff, col_diff = abs(row_idx - 3), abs(col_idx - 2)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 9:
                row_diff, col_diff = abs(row_idx - 3), abs(col_idx - 1)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 10:
                row_diff, col_diff = abs(row_idx - 3), abs(col_idx - 0)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 11:
                row_diff, col_diff = abs(row_idx - 2), abs(col_idx - 0)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 12:
                row_diff, col_diff = abs(row_idx - 1), abs(col_idx - 0)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 13:
                row_diff, col_diff = abs(row_idx - 1), abs(col_idx - 1)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 14:
                row_diff, col_diff = abs(row_idx - 1), abs(col_idx - 2)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)
            elif col == 15:
                row_diff, col_diff = abs(row_idx - 2), abs(col_idx - 2)
                d_dist = min(row_diff, col_diff)
                total_dist += (1.5 * d_dist) + abs(row_diff-col_diff)

    return total_dist


def astar(node, h_type, viewlog=False):
    """Return the first node encountered that equals the goal test using A* algorithm
    Return None if it does not find any,
    Return the number of nodes that expanded by the algorithm
    Return the maximum number of nodes that stored in memory

    :param h_type: str
    :param node: Node class
    :param viewlog: bool
    :return: Node class or None
    """
    frontier = []
    heapq.heappush(frontier, node)

    explored = {}
    num_of_exp_nodes = 0
    max_stored_nodes = 0

    while not len(frontier) == 0:
        if max_stored_nodes < len(frontier):
            max_stored_nodes = len(frontier)

        exp_node = heapq.heappop(frontier)
        num_of_exp_nodes += 1

        # This block is only for observing the current status in terminal
        if num_of_exp_nodes % 5000 == 0 and viewlog:
            print("Current # of expanded nodes : " + str(num_of_exp_nodes))

        if goal_test(exp_node.state):
            return exp_node, num_of_exp_nodes, max_stored_nodes

        for action in get_actions(exp_node.state):
            # Children are created with specific heuristic function (h_type)
            child_node = get_child(exp_node, action, h_type)
            if child_node.state_text not in explored or explored[child_node.state_text] > child_node.f:
                explored[child_node.state_text] = child_node.f
                heapq.heappush(frontier, child_node)
    else:
        # When frontier gets empty, the algorithm returns None as a result
        return None, num_of_exp_nodes, max_stored_nodes


def ils(node, limit, viewlog=False):
    """Return the first node encountered that equals the goal test using Iterative Lengthening algorithm
    Calls uniform cost search algorithm with a limit (recursive_ils),
    Return the number of nodes that expanded by the algorithm

    :param node: Node class
    :param limit: int
    :param viewlog: bool
    :return: Node class or None, int
    """
    n_limit = 0
    result = None
    num_of_exp_nodes = 0
    max_stored_nodes = 0

    # Calls uniform cost search with a limit (recursive_ils) iteratively
    while n_limit < limit:
        result, noen, msn = recursive_ils(node, n_limit, viewlog)

        if max_stored_nodes < msn:
            max_stored_nodes = msn

        if isinstance(result, str):
            n_limit += 0.5
            num_of_exp_nodes += noen
            noen = 0
            continue
        else: break

    return result, num_of_exp_nodes, max_stored_nodes


def recursive_ils(node, limit, viewlog):
    frontier = []
    heapq.heappush(frontier, node)

    explored = {}
    num_of_exp_nodes = 0
    max_stored_nodes = 0

    while not len(frontier) == 0:
        if max_stored_nodes < len(frontier):
            max_stored_nodes = len(frontier)

        exp_node = heapq.heappop(frontier)
        num_of_exp_nodes += 1

        # If the expanded node's cost is greater than limit, the function sends a Cutoff signal to its caller (ils)
        if exp_node.path_cost > limit:
            return "Cutoff", num_of_exp_nodes, max_stored_nodes

        # This block is only for observing the current status in terminal
        if num_of_exp_nodes % 5000 == 0 and viewlog:
            print("Current # of expanded nodes : " + str(num_of_exp_nodes))
            print("Current cost: " + str(exp_node.path_cost))

        if goal_test(exp_node.state):
            return exp_node, num_of_exp_nodes, max_stored_nodes

        for action in get_actions(exp_node.state):
            child_node = get_child(exp_node, action)
            if child_node.state_text not in explored or explored[child_node.state_text] > child_node.f:
                explored[child_node.state_text] = child_node.f
                heapq.heappush(frontier, child_node)

    else:
        # When frontier gets empty, the algorithm returns None as a result
        return None, num_of_exp_nodes, max_stored_nodes


def ucs(node, viewlog=False):
    """Return the first node encountered that equals the goal test using Uniform-Cost-Search algorithm,
    Return the number of nodes that expanded by the algorithm
    Return the maximum number of nodes that stored in memory

    :param node: Node()
    :param viewlog: bool
    :return: Node() or None, int, int
    """
    frontier = []
    heapq.heappush(frontier, node)

    explored = {}
    num_of_exp_nodes = 0
    max_stored_nodes = 0

    while not len(frontier) == 0:
        if max_stored_nodes < len(frontier):
            max_stored_nodes = len(frontier)

        exp_node = heapq.heappop(frontier)
        num_of_exp_nodes += 1

        # This block is only for observing the current status in terminal
        if num_of_exp_nodes % 5000 == 0 and viewlog:
            print("Current # of expanded nodes : " + str(num_of_exp_nodes))
            print("Current cost: "+str(exp_node.path_cost))

        if goal_test(exp_node.state):
            return exp_node, num_of_exp_nodes, max_stored_nodes

        for action in get_actions(exp_node.state):
            child_node = get_child(exp_node, action)
            if child_node.state_text not in explored or explored[child_node.state_text] > child_node.f:
                explored[child_node.state_text] = child_node.f
                heapq.heappush(frontier, child_node)

    else:
        # When frontier gets empty, the algorithm returns None as a result
        return None, num_of_exp_nodes, max_stored_nodes


def report_path(final_node, viewlog=True):
    """Return the solution's depth, Print the path and report for given goal Node

    :param final_node: Node()
    :return: int
    """
    if final_node:
        traverse = final_node
        path_ordered = []

        while traverse is not None:
            path_ordered.append(traverse)
            traverse = traverse.parent

        path_ordered.reverse()
        if viewlog:
            for n in path_ordered:
                print(n.action)
                print(np.matrix(n.state))
                print("Current cost:" + str(n.path_cost))
                print("")
            print("Cost:" + str(final_node.path_cost))

        # Returns solution's depth
        return len(path_ordered)-1
    else:
        print("Solution cannot be found!")
        return 0


def calculate_bfactor(depth, numof_en):
    """ Return the effective branching factor using Newton-Raphson method

    :param depth: int
    :param numof_en: int
    :return: int
    """
    np.seterr(over="ignore")
    numof_en = -1 * numof_en
    coeff = []

    for each in range(depth):
        coeff.append(1)

    coeff.append(numof_en)

    pol_func = np.poly1d(coeff)
    pol_der = np.polyder(pol_func)

    x0 = 2
    x = 0
    limit = 100
    i = 0

    while True:
        if i > limit:
            print("Solution cannot be found")
            x = 0
            break

        x = x0 - (np.polyval(pol_func, x0)/np.polyval(pol_der, x0))

        if abs(x-x0) < 1.48e-08:
            break

        i += 1
        x0 = x

    return x


if __name__ == '__main__':
    main()
