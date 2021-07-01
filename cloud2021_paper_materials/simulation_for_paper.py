#!/usr/bin/python3

import numpy as np
import random
import pandas as pd
import importlib.machinery
loader = importlib.machinery.SourceFileLoader('ALG', '../partitioning_alg/da_ALG.py')
ALG = loader.load_module()
loader = importlib.machinery.SourceFileLoader('OPT','../partitioning_alg/OPT.py')
OPT = loader.load_module()
loader = importlib.machinery.SourceFileLoader('baruah','../partitioning_alg/baruah.py')
baruah = loader.load_module()
loader = importlib.machinery.SourceFileLoader('heur','../partitioning_alg/first_fit.py')
heur = loader.load_module()

import seaborn as sns
import matplotlib.pyplot as plt
import time

TASK_COUNT=[10, 50, 100, 500, 1000] 	# depcited as 'n' in the paper
CPU_COUNT=100				# depicted as 'm' in the paper
SIMULATION_COUNT=2

def get_random_M(n, m):
    # number of items
    n = n

    # number of bins
    m = m

    # size-bin matrix M'
    M = np.random.randint(2, size=(n, m))
    # print(M)
    # delete all-zero columns
    idx = np.argwhere(np.all(M[..., :] == 0, axis=0))
    for i in idx:
        ones = random.randint(1, n)
        rows = list(range(n))
        for j in range(ones):
            pos = random.choice(rows)
            M[pos, i] = 1
            rows.remove(pos)

    # delete all-zero rows
    idx = np.argwhere(np.all(M[:, ...] == 0, axis=1))
    for i in idx:
        bin = random.randint(1, m)
        M[i, bin] = 1
    return M


def get_arbitrary_M():
    # Arbitrary test matrix
    # M=np.array([[1,0,0,1,0,0], [1,0,1,0,0,0], [1,0,1,0,1,1], [0,1,1,1,0,0], [0,1,0,0,1,0], [0,1,0,0,0,1]])
    # M= np.array([[0,1,0,0,1,0], [1,1,1,1,0,1], [1,0,1,0,0,1], [0,0,0,1,0,1], [1,0,1,1,1,1], [0,0,1,0,0,0]])
    M = np.array(
        [[0.43, 0.5, 0.35, 0.43, 0.5, 0.35], [0.33, 0.5, 0.25, 0.33, 0.6, 0.25], [0.33, 0.6, 0.25, 0.43, 0.6, 0.25],
         [0.43, 0.6, 0.35, 0.33, 0.6, 0.25], [0.33, 0.6, 0.25, 0.33, 0.5, 0.25],
         [0.43, 0.6, 0.25, 0.33, 0.5, 0.25]])
    return M


def get_utilization_matrix(n, m, one_count):
    # size-bin matrix M'
    M = np.random.randint(1, size=(n, m))

    possible_choices = []
    for i in range(n):
        for j in range(m):
            possible_choices.append((i, j))

    for item in range(n):
        random_bin = np.random.randint(0, m)
        M[item, random_bin] = 1
        possible_choices.remove((item, random_bin))

    for i in range(one_count - n):
        indexes = random.choice(possible_choices)
        M[indexes[0], indexes[1]] = 1

    # transform M to utilization matrix
    task_count, core_count = M.shape
    # M_util = M.copy()
    M_util = np.array(list(M[:, :]), dtype=float)

    for task in range(n):
        for cpu in range(m):
            fit_count = 0
            for i in M[:, cpu]:
                if i == 1:
                    fit_count += 1
            try:
                fit_value = 1 / fit_count
                if fit_value == 1.0:
                    not_fit_value = 1.1
                else:
                    not_fit_value = np.random.uniform(fit_value + 0.05, 1)

            except ZeroDivisionError:
                fit_value = 1.1
                not_fit_value = 1.1

            if M[task, cpu] == 1:
                M_util[task, cpu] = fit_value
            else:
                M_util[task, cpu] = not_fit_value

    return M_util


def get_homogen_utilization_matrix(n, m):
    utils = []
    first_util = np.random.uniform(0.1, 1)
    utils.append(first_util)
    sum = first_util
    for i in range(n - 1):
        util = np.random.uniform(0, min(m - sum, 1))
        utils.append(util)
        sum += util

    # size-bin matrix M'
    M = np.empty([n, m], dtype=float)

    for row in range(n):
        for column in range(m):
            M[row][column] = utils[row]

    # print(M)

    return M


def get_full_random_utilization_matrix(n, m):
    # size-bin matrix M'
    M = np.empty([n, m], dtype=float)

    for row in range(n):
        for column in range(m):
            M[row][column] = np.random.uniform(0.01, 1)

    return M


alg1_solutions = []
alg2_solutions = []
ff_solutions = []
opt_solutions = []

for sim in range(0, 1):
    sim_case = 0
    # for task_count in [10, 50, 100]:
    for task_count in TASK_COUNT:
        task_count=int(task_count)
        bin_count = CPU_COUNT
        one_count_range = range(int(task_count), int(task_count*11), int(task_count/2))
        for one_count in one_count_range:

            if one_count < task_count * bin_count and one_count >= task_count:
                print("ONE count: {} ---------------------------".format(one_count))
                print("{} - Simulation case {}, tasks: {}, ones:{}".format(time.asctime(), sim_case, task_count,
                                                                           one_count))
                df = pd.DataFrame(columns=['Sim. case', 'OPT', 'ALG1', 'ALG2', "matrix", "one count"])
                for simulation in range(SIMULATION_COUNT):
                    # print("{} - Simulation case {}, tasks: {}, ones:{}".format(time.asctime(), sim_case, task_count,
                    #                                                           one_count))
                    # M = get_random_M(6,6)
                    # M = get_arbitrary_M()

                    M = get_utilization_matrix(task_count, bin_count, one_count)
                    #M = get_homogen_utilization_matrix(task_count, bin_count)
                    # M = get_full_random_utilization_matrix(task_count, bin_count)

                    output_alg1 = False
                    output_alg2 = False
                    output_baruah = False
                    for i in range(1):
                        t1 = time.time()
                        alg1_solution = ALG.ALG(M, 1)
                        t2 = time.time()
                        alg1_time = t2-t1
                        t1 = time.time()
                        alg2_solution = ALG.ALG(M, 3)
                        t2 = time.time()
                        alg2_time = t2 - t1
                        # print("\t\tALG2 finished in {} sec".format(t2 - t1))
                        t1 = time.time()
                        baruah_solution = baruah.baruah_algorithm(M)
                        t2 = time.time()
                        br_time = t2 - t1
                        t1 = time.time()
                        developed_baruah_solution = baruah.developed_baruah_algorithm(M)
                        t2 = time.time()
                        dbr_time = t2 - t1
                        t1 = time.time()
                        ffd_solution = heur.first_fit_decreasing_ALG(M)
                        t2 = time.time()
                        ffd_time = t2 - t1
                        # print("\t\tFFD finished in {} sec".format(t2 - t1))
                        opt_solution = 1
                        #t1 = time.time()
                        #opt_solution, comment = OPT.OPT(M)
                        #t2 = time.time()
                        opt_time = t2 - t1
                        #print("\t\tOPT finished in {} sec".format(t2 - t1))

                        df = df.append(
                            {"Sim. case": sim_case, "ALG1": alg1_solution, "ALG2": alg1_solution,
                             "FFD": ffd_solution,
                             "BR": baruah_solution[0],
                             "DBR": developed_baruah_solution[0],
                             "OPT": opt_solution,
                             "ALG1 approximation rate": alg1_solution / opt_solution,
                             "ALG2 approximation rate": alg2_solution / opt_solution,
                             "FFD approximation rate": ffd_solution / opt_solution,
                             "BR approximation rate": baruah_solution[0] / opt_solution,
                             "DBR approximation rate": developed_baruah_solution[0] / opt_solution,
                             "ALG1 time":alg1_time,
                             "ALG2 time": alg2_time,
                             "BR time": br_time,
                             "DBR time": dbr_time,
                             "FFD time": ffd_time,
                             "OPT time": opt_time,
                             "matrix": M,
                             "one count": one_count, "task_count": task_count, "bin_count": bin_count},
                            ignore_index=True)
                        print("#Sim \tALG1 \tALG2 \tBR \tEBR \tFFD \tOPT")
                        print("{}. \t{}\t {}\t{} \t{} \t{} \t{}\n".format(simulation, alg1_solution, alg2_solution, baruah_solution,
                                                                    developed_baruah_solution,
                                                                    ffd_solution, opt_solution))
                    sim_case += 1

                print(df)
                print("Simulation results is written into the file 'heterogeneous_cluster_tests_matrix_{}x{}_{}.csv'\n".format(task_count, bin_count, one_count))
                df.to_csv(r'./heterogeneous_cluster_tests_matrix_{}x{}_{}.csv'.format(task_count, bin_count, one_count), index=False,
                          header=True)
