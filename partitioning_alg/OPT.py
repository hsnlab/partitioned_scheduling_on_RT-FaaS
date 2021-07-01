from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
import random
from pulp import PULP_CBC_CMD
import numpy as np

# Press the green button in the gutter to run the script.
def OPT(wcet):

    task_count, core_count = wcet.shape
    """
    
    wcet = dict()
    for task in range(task_count):
        for cpu in range(core_count):
            fit_count = 0
            #print(M)
            for i in M[:,cpu]:
                if i == 1:
                    fit_count += 1
            fit_value = 1/fit_count
            not_fit_value = fit_value + 0.1

            if M[task,cpu] == 1:
                wcet[(task, cpu)] = fit_value
            else:
                wcet[(task, cpu)] = not_fit_value
    """


    # Create the model
    model = LpProblem(name="partition-partition-scheduler", sense=LpMinimize)

    # Initialize the decision variables
    x_variables = dict()
    for t in range(task_count):
        for j in range(core_count):
            if (t, j) not in x_variables:
                x_variables[(t, j)] = LpVariable(name="x_task{}_cpu{}".format(t, j), lowBound=0, cat="Binary")
    y_variables = dict()
    for j in range(core_count):
        if j not in y_variables:
            y_variables[j] = LpVariable(name="y_{}".format(j), lowBound=0, cat="Binary")

    # Add the constraints to the model
    for t in range(task_count):
        model += (lpSum([x_variables[(t, j)] for j in range(core_count)]) == 1, "One_assignment_constraint{}".format(t))

    for j in range(core_count):
        model += (
            lpSum([x_variables[(t, j)] * wcet[t,j]  for t in range(task_count)]) <= 1 * y_variables[j],
            "EDF constraint on CPU{}".format(j))

    # Add the objective function to the model
    obj_func = lpSum(y_variables)
    model += obj_func

    # The problem data is written to an .lp file
    model.writeLP("TestProblem.lp")

    # Solve the problem
    #status = model.solve()
    status = None
    time_limit = 900
    status = model.solve(PULP_CBC_CMD(msg=0, timeLimit=time_limit, threads=1))

    #print("-------------------------------------------------------")
    #print(f"status: {model.status}, {LpStatus[model.status]}")
    #print(f"objective: {model.objective.value()}")

    #for var in model.variables():
    #    print(f"{var.name}: {var.value()}")

    #for name, constraint in model.constraints.items():
    #    print(f"{name}: {constraint.value()}")

    #print("Solver: {}".format(model.solver))

    if model.solutionTime > time_limit:
        return np.inf, LpStatus[model.status]
    elif status == 1:
        return model.objective.value(), LpStatus[model.status]
    else:
        #print("OPT is not optimal solution! Reason: {}".format(LpStatus[model.status]))
        return np.inf, LpStatus[model.status]
