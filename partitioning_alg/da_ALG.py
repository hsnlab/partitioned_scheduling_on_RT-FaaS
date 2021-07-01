import numpy as np
import random
import pandas as pd



def ALG(M, type):
    # Steps of ALG:

    # We skip step 1 and 2 since we initially deal with  binary matrix
    def column_a_greater_b(a, b):
        delete_column_a = True
        num_items = len(a)
        for elem_index in range(num_items):
            if a[elem_index] == 1 and b[elem_index] != 1:
                return False
        return delete_column_a

    def step_6(matrix):
        columns_to_delete = []
        num_items, num_bins = matrix.shape

        matrix["sum"] = matrix.sum(axis=1)
        #print(matrix)

        for i in range(num_items):
            if matrix["sum"][i] == 1:
                for column in matrix.columns:
                    if matrix[column][i] == 1:
                        columns_to_delete.append(column)
                        break

        columns_to_delete = list(set(columns_to_delete))
        return columns_to_delete

    def step_3(M):
        step_3_finish = False
        while not step_3_finish:
            #print(M)
            num_items, num_bins = M.shape
            columns_to_delete = []
            step_3_finish = True
            for column_i in M.columns:
                for column_j in M.columns:
                    if column_i != column_j:
                        if column_a_greater_b(M[column_i], M[column_j]):
                            columns_to_delete.append(column_i)
                if columns_to_delete != []:
                    M = M.drop(columns_to_delete, axis=1)
                    step_3_finish = False
                    break

        #print(M)
        return M

    def step_4(M):

        sorted_columns = []
        for column in M.columns:
            weight = 0
            for j in M[column]:
                if j == 1:
                    weight += 1

            if sorted_columns == []:
                sorted_columns.append({"weight":weight, "column":column})
            else:
                inserted = False
                for j in sorted_columns:
                    if weight >= j["weight"]:
                        sorted_columns.insert(sorted_columns.index(j), {"weight":weight, "column":column})
                        inserted = True
                        break
                if not inserted:
                    sorted_columns.append({"weight":weight, "column":column})

        return sorted_columns

    def step_2(M):
        columns_count, rows_count = M.shape
        M_boolean = pd.DataFrame(0, index=M.index, columns=M.columns)

        """
        for c in M.columns:
            until_index = 0
            for i in range(1, columns_count + 1):
                sum_value = M.nsmallest(i,c)[c].sum()
                if sum_value > 1:
                    one_indexes = M.nsmallest(until_index, c).index
                    M_boolean.loc[one_indexes, c] = 1
                    break
                else:
                    until_index = i

        """
        for c in M.columns:
            #column = M[[c]]
            ordered = M.nsmallest(rows_count,c)
            until_index = 0
            for i in range(1,columns_count+1):
                sum_value = ordered[c].head(i).sum()
                if sum_value > 1:
                    break
                else:
                    until_index = i
            one_indexes = ordered[c].index[:until_index]
            M_boolean.loc[one_indexes, c] = 1

        return M_boolean

    num_items, num_bins = M.shape
    rows_to_delete = []
    columns_to_delete = []
    unassigned_items = ["item_{}".format(i) for i in range(num_items)]
    assignments = {}
    matrix_original = pd.DataFrame(M, columns=["bin_{}".format(i) for i in range(num_bins)],
                              index=["item_{}".format(i) for i in range(num_items)])
    for item in matrix_original.index:
        assignments[item] = None

    while unassigned_items != []:

        try:
            matrix = matrix.drop(['sum'], axis=1)
        except:
            pass

        if columns_to_delete != []:
            rows_to_delete = []
            for bin in columns_to_delete:
                for item in matrix[bin].index:
                    if matrix[bin][item] == 1:
                        if assignments[item] is None:
                            assignments[item] = bin
                            unassigned_items.remove(item)
                            rows_to_delete.append(item)

            matrix_original = matrix_original.drop(columns_to_delete, axis=1)
            matrix_original = matrix_original.drop(rows_to_delete, axis=0)


        #print(matrix)
        if unassigned_items!= []:

            if all("bin" not in i for i in matrix_original.columns):
                #raise Exception("Unassigned item(s):{} exist without possible bin to put!".format(unassigned_items))
                #print("Unassigned item(s):{} exist without possible bin to put!".format(unassigned_items))
                return np.inf

            # Step 2: Convert to boolean matrix
            matrix = step_2(matrix_original)

            # Step 4: Calculate weights
            sorted_columns = step_4(matrix)

            if type != 3:
                # Step 6: Visiting bins with exclusive items
                columns_to_delete = step_6(matrix)

                if columns_to_delete != []:
                    if type == 1:
                        for c in sorted_columns:
                            if c['column'] in columns_to_delete:
                                columns_to_delete = [c['column']]
                                break
                    elif type == 2:
                        decreasing = sorted_columns.copy()
                        while True:
                            if decreasing[-1]['column'] in columns_to_delete:
                                columns_to_delete = [decreasing[-1]['column']]
                                break
                            decreasing = decreasing[:-1]
            else:
                columns_to_delete = []


            if columns_to_delete == []:

                # Step 7
                columns_with_highes_weights = []
                columns_with_highes_weights.append(sorted_columns[0])
                for i in sorted_columns[1:]:
                    if i["weight"] == sorted_columns[0]["weight"]:
                        columns_with_highes_weights.append(i)
                    else:
                        break
                selected_bin = pos = random.choice(columns_with_highes_weights)
                columns_to_delete.append(selected_bin["column"])


            # Step 3: Filtering best bins
            #matrix = step_3(matrix)



    #print("Assignment: {}".format(assignments))
    bins = []
    for key, value in assignments.items():
        if value not in bins:
            bins.append(value)
    #print("ALG(I) = {}".format(len(bins)))
    return len(bins)