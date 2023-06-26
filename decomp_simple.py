from gurobipy import Model, GRB, read, LinExpr, disposeDefaultEnv
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import time
import os
import pandas as pd

import importlib

def read_model(instance):
    m = read('collection/'+instance+'.mps.gz')
    m_new = read('collection/'+instance+'.mps.gz')
    m_new1 = read('collection/'+instance+'.mps.gz')
    m_new2 = read('collection/'+instance+'.mps.gz')
    return m, m_new, m_new1, m_new2

def get_info(m):
    A = m.getA()
    #print(A)
    # plot sparsity of A
    #plt.spy(A,markersize=0.5)
    #plt.show()
    x = m.getVars()
    for i in range(len(x)):
        x[i].VarName = 'x_'+str(i)
    con = m.getConstrs()
    #print(len(x),len(con))
    sizeA = A.get_shape()
    #print(sizeA)
    A.eliminate_zeros()
    nonzeros = A.nonzero()
    
    # get rhs of orgininal problem
    RHS = m.getAttr('RHS')
    SENSE = m.getAttr('Sense')
    # Get the variables in the model
    vars = m.getVars()
    
    return A, x, con, sizeA, nonzeros, RHS, SENSE, vars
    
# def solve_LP():

def decomp_model(A, sizeA, con, nonzeros, vars, HGtype, instance, nBlocks):
    if HGtype == 'r':
        print("Create Row-Net Hypergraph")
        # Create Row-Net Hypergraph
        rNetHg = []
        for i in range(sizeA[0]):
            rNetHg.append(A.getrow(i).nonzero()[1])
        nNodes = sizeA[1]+round(len(nonzeros)*0.2)
        nHedges = sizeA[0]
        wrtStr = str(nHedges)+'\t'+str(nNodes)+'\n'
        for i in range(nHedges):
            for k in range(len(rNetHg[i])):
                wrtStr += str(rNetHg[i][k]+1)
                if k < len(rNetHg[i])-1:
                    wrtStr += '\t'
            wrtStr += '\n'
            
        print(os.getcwd())
        os.chdir('Tests/readMPS/')
        f = open('HGraphFiles/'+instance+'rHG','w')
        f.write(wrtStr)
        f.close


        # Use hmetis to decompose the constraint matrix
        os.chdir('../../hmetis-1.5-osx-i686/')
        #os.chdir('../../hmetis-1.5-linux/')

        open('../Tests/readMPS/HGraphFiles/'+instance+'rHG')
        print('shmetis ../Tests/readMPS/HGraphFiles/'+instance+'rHG '+str(nBlocks)+' 1')
        print("Current directory:", os.getcwd())
        #os.system('/bin/bash shmetis ../Tests/readMPS/HGraphFiles/'+instance+'rHG '+str(nBlocks)+' 1')
        if not os.path.exists('HGraphFiles/'+instance+'rHG.part.'+str(nBlocks)):
            os.system('./shmetis ../Tests/readMPS/HGraphFiles/'+instance+'rHG '+str(nBlocks)+' 1')
        else:
            print("Hypergraph exist!!")

        # Read and plot the reordered constraint matrix
        #os.chdir('../')
        os.chdir('../Tests/readMPS/')
        print(os.getcwd())
        f = open('HGraphFiles/'+instance+'rHG.part.'+str(nBlocks))
        xx = f.readlines()
        f.close
        colgroup = {}
        rowToGroup = {}
        for i in range(nBlocks+1):
            colgroup[i] = []

        for i in range(sizeA[1]):
            colgroup[int(xx[i].strip('\n'))].append(i)
            rowToGroup[i] = int(xx[i].strip('\n'))

        varmap = {}
        ind = 0
        for i in range(nBlocks):
            for k in colgroup[i]:
                varmap[k] = ind
                ind += 1
        rowgroup = {}
        for i in range(nBlocks+1):
            rowgroup[i] = []

        for i in range(sizeA[0]):
            groupind = rowToGroup[A.getrow(i).nonzero()[1][0]]
            if all([rowToGroup[A.getrow(i).nonzero()[1][k]] == groupind for k in range(len(A.getrow(i).nonzero()[1]))]):
                rowgroup[groupind].append(i)
            else:
                rowgroup[nBlocks].append(i)

        rowmap = {}
        ind = 0
        for i in range(nBlocks+1):
            for k in rowgroup[i]:
                rowmap[k] = ind
                ind += 1
        A_coo = A.tocoo(copy=True)
        A_row = A_coo.row
        A_col = A_coo.col
        A_data = A_coo.data
        A_reord_row = [rowmap[i] for i in A_row]
        A_reord_col = [varmap[i] for i in A_col]
        # A_reord is only for plotting, its corresponding data is incorrect
        A_reord = csr_matrix((A_data,(A_reord_row,A_reord_col)))
        # plot
        plt.spy(A_reord,markersize=0.5)
        plt.show()
        os.chdir('../../')
    

    # Print the names of the variables
    #for v in vars:
        #print(v.varName)

    var_group = [x.strip() for x in xx]
    
    counter = {}
    
    for element in var_group:
        counter[element] = counter.get(element, 0) + 1
    
    element_counts = list(counter.items())
    element_counts.sort(key=lambda x: x[1])

    lowest_count = element_counts[0][1]
    highest_count = element_counts[-1][1]
    
    #get the row need to add slack variable
    add_slack = []
    for i in range(A.shape[0]):
        ct = []
        for n in A.getrow(i).nonzero()[1]:
            ct.append(var_group[n])
        #if i % 10 ==0:
            #print(np.unique(np.array(ct)))
        if len(np.unique(np.array(ct))) == 1:
            #print('no add')
            add_slack.append(True)
        else:
            #print('add')
            add_slack.append(False)
    
    num_linking_cons = np.sum(add_slack)
    
    return add_slack, num_linking_cons, lowest_count, highest_count

def solve_decomp_with_slack(A, vars, RHS, SENSE, add_slack):
    new_model = Model()
    new_model.modelSense = GRB.MINIMIZE
    new_model.update()
    
    # add variables to the model
    X_vars = [0 for i in range(A.shape[1])]
    for i in range(A.shape[1]):
        curVar = new_model.addVar(lb = vars[i].lb, ub = vars[i].ub, vtype = vars[i].vtype,
                                               name = "X" + str(i))
        X_vars[i] = curVar

    new_model.update()
    
    slack_vars = []
    for i in range(A.shape[0]):
        ConsExpr = LinExpr()

        for j in A.getrow(i).nonzero()[1]:
            ConsExpr += A[i,j]*X_vars[j]
        if add_slack[i] and SENSE[i] == "<":
            cur_slack = new_model.addVar(vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i))
            slack_vars.append(cur_slack)
            ConsExpr -= cur_slack
        elif add_slack[i] and SENSE[i] == ">":
            cur_slack = new_model_all.addVar(vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i))
            slack_vars_all.append(cur_slack)
            ConsExpr += cur_slack
        elif add_slack[i] and SENSE[i] == "=":
            cur_slack1 = new_model.addVar(vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i) + '_1')
            cur_slack2 = new_model.addVar(vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i) + '_2')
            slack_vars.append(cur_slack1)
            slack_vars.append(cur_slack2)
            ConsExpr += cur_slack1 - cur_slack2

        new_model.addConstr(lhs = ConsExpr, sense = SENSE[i], rhs = RHS[i], name = 'Constr'+str(i))  

    new_model.update()

    objExpr = LinExpr()

    for i in range(len(slack_vars)):
        objExpr += slack_vars[i]
    
    new_model.setObjective(objExpr, GRB.MINIMIZE)
    
    new_model.setParam(GRB.Param.Seed, 77)

    start_time = time.time()
    new_model.optimize()
    end_time = time.time()
    feasibility_time = end_time - start_time
    
    status = new_model.status
    
    print(new_model.getObjective().getValue())
    
    return feasibility_time, status

def solve_obj_0(m_new1):
    # set the obj to 0 for the original problem to get a feasible solution

    # Set a new objective function
    m_new1.setObjective(0, GRB.MINIMIZE)
    
    m_new1.setParam(GRB.Param.Seed, 77)

    start_time = time.time()

    # Optimize the modified model
    m_new1.optimize()

    end_time = time.time()

    obj_to_0_time = end_time - start_time
    
    status = m_new1.status
    
    print(m_new1.getObjective().getValue())
    
    return obj_to_0_time, status
    
    
def solve_all_add_slack(A, vars, SENSE, RHS):
    # all add slack variable
    new_model_all = Model()
    new_model_all.modelSense = GRB.MINIMIZE
    new_model_all.update()

    X_vars_all = [0 for i in range(A.shape[1])]
    for i in range(A.shape[1]):
        curVar = new_model_all.addVar(lb = vars[i].lb, ub = vars[i].ub, vtype = vars[i].vtype,
                                               name = "X" + str(i))
        X_vars_all[i] = curVar

    new_model_all.update()
        

    slack_vars_all = []
    for i in range(A.shape[0]):
        ConsExpr = LinExpr()

        for j in A.getrow(i).nonzero()[1]:
            ConsExpr += A[i,j]*X_vars_all[j]
        if SENSE[i] == "<":
            cur_slack = new_model_all.addVar(vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i))
            slack_vars_all.append(cur_slack)
            ConsExpr -= cur_slack
        elif SENSE[i] == ">":
            cur_slack = new_model_all.addVar(vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i))
            slack_vars_all.append(cur_slack)
            ConsExpr += cur_slack
        elif SENSE[i] == "=":
            cur_slack1 = new_model_all.addVar(vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i) + '_1')
            cur_slack2 = new_model_all.addVar(vtype = GRB.CONTINUOUS,
                                               name = "S" + str(i) + '_2')
            slack_vars_all.append(cur_slack1)
            slack_vars_all.append(cur_slack2)
            ConsExpr += cur_slack1 - cur_slack2
        
        new_model_all.addConstr(lhs = ConsExpr, sense = SENSE[i], rhs = RHS[i], name = 'Constr'+str(i))  

    new_model_all.update()

    objExpr = LinExpr()

    for i in range(len(slack_vars_all)):
        objExpr += slack_vars_all[i]
    
    new_model_all.setObjective(objExpr, GRB.MINIMIZE)
    
    new_model_all.setParam(GRB.Param.Seed, 77)

    start_time = time.time()
    new_model_all.optimize()
    end_time = time.time()

    all_add_slack_time = end_time - start_time
    
    status = new_model_all.status
    
    print(new_model_all.getObjective().getValue())
    
    return all_add_slack_time, status

    