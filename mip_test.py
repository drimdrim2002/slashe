from util import *
import json

from util import *
import gurobipy as gp
from gurobipy import GRB
import itertools
import numba
import numpy as np
from itertools import permutations, combinations
import math
import time
import concurrent.futures
import sys


def solve_mip_with_gurobi(feasible_bundles, ALL_AVA, K, timelimit, covering=False):
    print(f'solve mip with corvering :{covering}')
    mip_start_time = time.time()

    get_bdl_shop_seq = lambda b: b[0][0]
    get_bdl_dlv_seq = lambda b: b[0][1]
    get_bdl_rider = lambda b: b[1]
    get_bdl_dist = lambda b: b[2]
    get_bdl_vol = lambda b: b[3]
    get_bdl_cost = lambda b: b[4]

    bd_to_k = {k: [] for k in range(K)}
    bd_to_r = {r: [] for r in [0, 1, 2]}

    for idx, bdl in enumerate(feasible_bundles):
        for k in get_bdl_dlv_seq(bdl):
            bd_to_k[k].append(idx)

        r = get_bdl_rider(bdl)
        bd_to_r[r].append(idx)

    m = gp.Model("IP")

    x = m.addVars(
        len(feasible_bundles),
        obj=[get_bdl_cost(bd) / K for bd in feasible_bundles],
        vtype=GRB.BINARY,
        name='x'
    )

    for r in [0, 1, 2]:
        m.addConstr(gp.quicksum(x[idx] for idx in bd_to_r[r]) <= ALL_AVA[r], name=f'max_bundles_{r}')

    if covering:
        for k in range(K):
            m.addConstr(gp.quicksum(x[idx] for idx in bd_to_k[k]) >= 1, name=f'order_{k}_bundle')

    else:
        for k in range(K):
            m.addConstr(gp.quicksum(x[idx] for idx in bd_to_k[k]) == 1, name=f'order_{k}_bundle')

    remaining_time = timelimit - (time.time() - mip_start_time)
    m.setParam('TimeLimit', remaining_time)

    m.setParam(GRB.Param.Threads, 4)

    m.optimize()

    final_bundles = []
    if m.status == GRB.OPTIMAL or m.SolCount > 0:
        print('* solution found:')
        for bundle_idx, (v, bd) in enumerate(zip(m.getVars(), feasible_bundles)):
            if v.x > 0.5:
                final_bundles.append(feasible_bundles[bundle_idx])

        print(f'* Objective value: {m.objVal}')


if __name__ == '__main__':
    available_riders = np.array([10, 15, 50])
    K = 50
    file_name = 'all_feasible_bundles_K50_1.txt'
    with open(file_name, 'r') as file:
        all_feasible_bundles_json = json.load(file)

    all_feasible_bundles = []
    for bundle_json in all_feasible_bundles_json:
        shop_seq = np.array(bundle_json['shop_seq_list'])
        dlvry_seq = np.array(bundle_json['dlvry_seq_list'])
        rider_type = bundle_json['rider_type']
        dist = bundle_json['dist']
        vol = bundle_json['vol']
        cost = bundle_json['cost']
        status = bundle_json['status']

        tuple_line = ((shop_seq, dlvry_seq), rider_type, dist, vol, cost, status)
        all_feasible_bundles.append(tuple_line)

    solve_mip_with_gurobi(all_feasible_bundles, available_riders, K, 60, True)