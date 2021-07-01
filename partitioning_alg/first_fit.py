import numpy as np
import random
import pandas as pd

def first_fit_ALG(M):

    num_items, num_bins = M.shape
    matrix_original = pd.DataFrame(M, columns=["bin_{}".format(i) for i in range(num_bins)],
                                   index=["item_{}".format(i) for i in range(num_items)])

    assignment = {}
    for bin in matrix_original.columns:
        assignment[bin] = {"items":[], "free capacity": 1}
    for item in matrix_original.index:
        assigned = False
        for bin in matrix_original.columns:
            size = matrix_original[bin][item]
            if assignment[bin]["free capacity"] - size >= 0:
                assignment[bin]["items"].append(item)
                assignment[bin]["free capacity"] -= size
                assigned = True
                break
        if not assigned:
            #raise Exception("Item {} cannot be assign to any of the bins by the first fit algorithm".format(bin))
            #print("Item {} cannot be assign to any of the bins by the first fit algorithm".format(item))
            return float('inf')

    # return the number of used bins
    used_bins = 0
    for bin, assignments in assignment.items():
        if assignments["items"] != []:
            used_bins +=1

    #print(assignment)

    return used_bins


def first_fit_decreasing_ALG(M):

    def sortSecond(val):
        return val[1]

    num_items, num_bins = M.shape
    matrix_original = pd.DataFrame(M, columns=["bin_{}".format(i) for i in range(num_bins)],
                                   index=["item_{}".format(i) for i in range(num_items)])

    assignment = {}
    for bin in matrix_original.columns:
        assignment[bin] = {"items":[], "free capacity": 1}
    items = []
    for item in matrix_original.index:
        max_size = 0
        for bin in matrix_original.columns:
            if matrix_original[bin][item] > max_size:
                max_size = matrix_original[bin][item]
        items.append((item,max_size))
    items.sort(key = sortSecond, reverse = True)

    for i in items:
        item = i[0]
        assigned = False
        for bin in matrix_original.columns:
            size = matrix_original[bin][item]
            if assignment[bin]["free capacity"] - size >= 0:
                assignment[bin]["items"].append(item)
                assignment[bin]["free capacity"] -= size
                assigned = True
                break
        if not assigned:
            #raise Exception("Item {} cannot be assign to any of the bins by the first fit algorithm".format(bin))
            #print("Item {} cannot be assign to any of the bins by the first fit algorithm".format(item))
            return float('inf')

    # return the number of used bins
    used_bins = 0
    for bin, assignments in assignment.items():
        if assignments["items"] != []:
            used_bins +=1

    #print(assignment)

    return used_bins

def best_fit_decreasing_ALG(M):

    num_items, num_bins = M.shape
    matrix_original = pd.DataFrame(M, columns=["bin_{}".format(i) for i in range(num_bins)],
                                   index=["item_{}".format(i) for i in range(num_items)])

    assignment = {}
    for bin in matrix_original.columns:
        assignment[bin] = {"items": [], "free capacity": 1}
    for item in matrix_original.index:
        bin_with_max_load = None
        min_free_space = 100
        max_load = 1-min_free_space
        for bin in matrix_original.columns:
            size = matrix_original[bin][item]
            if assignment[bin]["free capacity"] - size <= min_free_space and assignment[bin]["free capacity"] - size >= 0:
                min_free_space = assignment[bin]["free capacity"] - size
                bin_with_max_load = bin

        if bin_with_max_load == None:
            #raise Exception("Item {} cannot be assign to any of the bins by the first fit algorithm".format(bin))
            #print("Item {} cannot be assign to any of the bins by the first fit algorithm".format(item))
            return float('inf')

        else:
            assignment[bin_with_max_load]["items"].append(item)
            assignment[bin_with_max_load]["free capacity"] -= matrix_original[bin_with_max_load][item]

    # return the number of used bins
    used_bins = 0
    for bin, assignments in assignment.items():
        if assignments["items"] != []:
            used_bins +=1

    #print(assignment)

    return used_bins


def worst_fit_decreasing_ALG(M):

    num_items, num_bins = M.shape
    matrix_original = pd.DataFrame(M, columns=["bin_{}".format(i) for i in range(num_bins)],
                                   index=["item_{}".format(i) for i in range(num_items)])

    assignment = {}
    for bin in matrix_original.columns:
        assignment[bin] = {"items": [], "free capacity": 1}
    for item in matrix_original.index:
        bin_with_min_load = None
        max_load = 0
        for bin in matrix_original.columns:
            size = matrix_original[bin][item]
            if assignment[bin]["free capacity"] - size >= max_load and assignment[bin]["free capacity"] - size <=1:
                max_load = assignment[bin]["free capacity"] - size
                bin_with_min_load = bin

        if bin_with_min_load == None:
            #raise Exception("Item {} cannot be assign to any of the bins by the first fit algorithm".format(bin))
            #print("Item {} cannot be assign to any of the bins by the first fit algorithm".format(item))
            return float('inf')

        else:
            assignment[bin_with_min_load]["items"].append(item)
            assignment[bin_with_min_load]["free capacity"] -= matrix_original[bin_with_min_load][item]

    # return the number of used bins
    used_bins = 0
    for bin, assignments in assignment.items():
        if assignments["items"] != []:
            used_bins +=1

    #print(assignment)

    return used_bins