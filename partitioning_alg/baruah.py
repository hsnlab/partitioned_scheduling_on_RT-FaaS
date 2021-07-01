from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
import random
from pulp import PULP_CBC_CMD
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

def developed_baruah_algorithm(wcet):
    task_count, core_count = wcet.shape
    column_sums_original = wcet.sum(axis=0)
    for i in range(core_count):
        column_sums = column_sums_original.copy()
        indexes = []
        for j in range(i+1):
            index = np.where(column_sums == min(column_sums))
            indexes.append(index[0][0])
            column_sums[index[0]] = 100000

        modified_wcet = wcet[:,indexes]
        try:
            solution, comment = baruah_algorithm(modified_wcet)
        except:
            solution = np.inf
            comment = "Exception during execution"
            pass
        if solution != np.inf:
            return solution, comment

    return solution, comment

# Press the green button in the gutter to run the script.
def baruah_algorithm(wcet):
    task_count, core_count = wcet.shape

    # Create the model
    model = LpProblem(name="LPR-feas", sense=LpMinimize)

    # Initialize the decision variables
    x_variables = dict()
    for t in range(task_count):
        for j in range(core_count):
            if (t, j) not in x_variables:
                x_variables[(t, j)] = LpVariable(name="x_task{}_cpu{}".format(t, j), lowBound=0)

    u_variable = LpVariable(name="U", lowBound=0)

    # Add the constraints to the model
    for t in range(task_count):
        model += (lpSum([x_variables[(t, j)] for j in range(core_count)]) == 1, "One_assignment_constraint{}".format(t))

    for j in range(core_count):
        model += (
            lpSum([x_variables[(t, j)] * wcet[t, j] for t in range(task_count)]) <= u_variable,
            "EDF constraint on CPU{}".format(j))

    # Add the objective function to the model
    obj_func = lpSum(u_variable)
    model += obj_func

    # The problem data is written to an .lp file
    # model.writeLP("Baruah_LP.lp")

    # Solve the problem
    # status = model.solve()
    status = model.solve(PULP_CBC_CMD(msg=0, timeLimit=900))

    #print("-------------------------------------------------------")
    #print(f"status: {model.status}, {LpStatus[model.status]}")
    #print(f"objective: {model.objective.value()}")

    # create bipartite graph
    B = nx.Graph()

    partitioning = dict()
    for t in range(task_count):
        B.add_nodes_from(["task{}".format(t)], bipartite=0)
        partitioning["task{}".format(t)] = None
    for j in range(core_count):
        B.add_nodes_from(["cpu{}".format(j)], bipartite=1)

    for var in model.variables():
        #print(f"{var.name}: {var.value()}")

        if var.name == "U":
            if var.value() > 1:
                #raise Exception("Invalid LP solution!")
                return np.inf, "Invalid LP solution!"

        else:
            if var.value() > 0:
                #print(var.name)
                string_parts = var.name.split("_")
                task = int(string_parts[1][4:])
                cpu = int(string_parts[2][3:])
                B.add_edges_from([("task{}".format(task), "cpu{}".format(cpu))])

    #X, Y = bipartite.sets(B)
    draw = False
    try:
        X, Y = bipartite.sets(B)
    except:
        draw = False
        #plt.close()
        #nx.draw(B, with_labels=True)
        #plt.show()

        X = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
        Y = set(B) - X

    #nx.draw(B, with_labels=True, pos = nx.drawing.layout.bipartite_layout(B, X) )
    #plt.show()
    #plt.close()

    #nx.draw(B, with_labels=True)
    #plt.show()

    # delete exact mappings' tasks
    task_vertices_to_delete = []
    for task in X:
        #print("Degree of {} is {}".format(task,B.degree[task]))
        if B.degree[task] == 1:
            task_vertices_to_delete.append(task)
            partitioning[task] = list(B.neighbors(task))[0]
    for task in task_vertices_to_delete:
        B.remove_node(task)

    #plt.close()

    #if draw:
    #    nx.draw(B, with_labels=True)
    #    plt.show()

    # TODO: find cycles

    # create bipartite graph
    """
    B = nx.Graph()
    B.add_nodes_from(["task0"], bipartite=0)
    B.add_nodes_from(["cpu0"], bipartite=1)
    B.add_nodes_from(["task1"], bipartite=0)
    B.add_nodes_from(["cpu1"], bipartite=1)
    B.add_edges_from([("task0", "cpu0")])
    B.add_edges_from([("task0", "cpu1")])
    B.add_edges_from([("task1", "cpu0")])
    B.add_edges_from([("task1", "cpu1")])
    """

    cycle = False
    try:
        cycle = list(nx.find_cycle(B, orientation="ignore"))
        print(cycle)
        cycle = True
    except:
        pass

    if cycle:
        raise Exception("Cycle is found")

    #unassigned_tasks, cpus = bipartite.sets(B)
    unassigned_tasks = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    cpus = set(B) - unassigned_tasks
    while len(unassigned_tasks) != 0:

        for t, c in partitioning.items():
            if c != None:
                try:
                    B.remove_node(t)
                except nx.exception.NetworkXError:
                    pass

        root = None
        tasks = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
        cpus = set(B) - tasks
        for task in tasks:
            if B.degree[task] == 1:
                root = task
        if root == None:
            root_cpu = random.choice(list(cpus))
            B.add_node("arbitrary_task", bipartite=0)
            B.add_edges_from([("arbitrary_task", root_cpu)])
            root = "arbitrary_task"
            unassigned_tasks.add("arbitrary_task")

        #if draw:
        #    plt.close()
        #    nx.draw(B, with_labels=True)
        #    plt.show()

        visited_nodes = [root]
        unvisited_nodes = []
        task_cpus = B.neighbors(root)
        task_cpus = list(task_cpus)
        unvisited_nodes.extend(task_cpus)
        unvisited_nodes = list(set(unvisited_nodes))
        mapped_cpu = random.choice(task_cpus)
        #print("task {} -> cpu {}".format(root,mapped_cpu))
        partitioning[root] = mapped_cpu
        unassigned_tasks.remove(root)
        while unvisited_nodes != []:
            start_node = unvisited_nodes[0]
            visited_nodes.append(start_node)
            unvisited_nodes.remove(start_node)
            neigh = B.neighbors(start_node)
            for node in neigh:
                if node not in visited_nodes:
                    unvisited_nodes.append(node)

            if "task" in start_node:
                task_cpus = B.neighbors(start_node)
                task_cpus = [i for i in task_cpus if i not in visited_nodes]
                mapped_cpu = random.choice(task_cpus)
                #print("task {} -> cpu {}".format(start_node, mapped_cpu))
                partitioning[start_node] = mapped_cpu
                unassigned_tasks.remove(start_node)

    #for name, constraint in model.constraints.items():
    #    print(f"{name}: {constraint.value()}")

    #print("Solver: {}".format(model.solver))

    if status == 1:
        used_cpus = []
        try:
            partitioning.pop("arbitrary_task")
        except KeyError:
            pass
        #check if partitioning valid
        for task, cpu in partitioning.items():
            if cpu == None:
                raise Exception("There is a no mapping")
            used_cpus.append(cpu)
        for cpu_iter in range(core_count):
            cpu = "cpu{}".format(cpu_iter)

            sum_utilization = 0
            for task, cpu_ in partitioning.items():
                if cpu == cpu_:
                    sum_utilization += wcet[int(task[4:])][cpu_iter]
            if sum_utilization > 1.0001:
                return np.inf, "Overloaded cpu '{}': {}".format(cpu, sum_utilization)
        used_cpus = list(set(used_cpus))
        return len(used_cpus), "OK"

    else:
        # print("OPT is not optimal solution! Reason: {}".format(LpStatus[model.status]))
        return np.inf, LpStatus[model.status]
