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


def solve(K, all_orders, all_riders, dist_mat, timelimit=60):
    start_time = time.time()

    has_trace = hasattr(sys, 'gettrace') and sys.gettrace() is not None
    has_breakpoint = sys.breakpointhook.__module__ != "sys"
    isdebug = has_trace or has_breakpoint
    print(f"{has_trace=} {has_breakpoint=} {isdebug=}")

    timelimit = 600 if isdebug else 60

    # A solution is a list of bundles
    solution = []

    # K 및 time limit에 따라서 아래 parameter를 정한다.
    conditions = get_conditions()

    # index3: 3개 bundle을 만들 때 얼마나 많은 후보를 탐색하는지
    # index4: 4개 bundle을 만들 때 얼마나 많은 후보를 탐색하는지
    # index5: 5개 bundle을 만들 때 얼마나 많은 후보를 탐색하는지
    # index6: 6개 bundle을 만들 때 얼마나 많은 후보를 탐색하는지
    # time_limit2: 2개 bundle을 만들 때 제한 시간
    # time_limit3: 3개 bundle을 만들 때 제한 시간
    # time_limit4: 4개 bundle을 만들 때 제한 시간
    # time_limit5: 5개 bundle을 만들 때 제한 시간
    # time_limit6: 6개 bundle을 만들 때 제한 시간
    # time_limit6: 6개 bundle을 만들 때 제한 시간
    # max_2bundles: 2개 bundle의 최대 크기
    # slack : 60초 time out을 피하기 위한 slack (slack이 5초면 55초에 종료 시작)

    default_params = {
        "index3": 1, "index4": 1, "index5": 1, "index6": 1,
        "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 2, "time_limit6": 2,
        "max_2bundles": 100000, "slack": 5
    }

    params = default_params

    for condition_dict in conditions:
        if condition_dict["condition"](K, timelimit):
            params = condition_dict["params"]
            break

    # debugging을 편하게 하기 위한 처리
    if isdebug:
        params['time_limit2'] = 9999
        params['time_limit3'] = 9999
        params['time_limit4'] = 9999
        params['time_limit5'] = 9999
        params['time_limit6'] = 9999

    if "max_2bundles" not in params:
        params["max_2bundles"] = default_params["max_2bundles"]

    index3, index4, index5, index6 = params["index3"], params["index4"], params["index5"], params["index6"]
    time_limit2, time_limit3, time_limit4, time_limit5, time_limit6 = params["time_limit2"], params["time_limit3"], \
        params["time_limit4"], params["time_limit5"], params["time_limit6"]
    max_2bundles, slack = params["max_2bundles"], params["slack"]

    for r in all_riders:
        r.T = np.round(dist_mat / r.speed + r.service_time)
        if r.type == 'BIKE':
            T_B = r.T
            VC_B = r.var_cost
            FC_B = r.fixed_cost
            CAPA_B = r.capa
            AVA_B = r.available_number

        elif r.type == 'WALK':
            T_W = r.T
            VC_W = r.var_cost
            FC_W = r.fixed_cost
            CAPA_W = r.capa
            AVA_W = r.available_number

        elif r.type == 'CAR':
            T_C = r.T
            VC_C = r.var_cost
            FC_C = r.fixed_cost
            CAPA_C = r.capa
            AVA_C = r.available_number

    # ALL_T[0] = T_B, ALL_T[1] = T_W, ALL_T[2] = T_C
    ALL_T = np.stack([T_B, T_W, T_C])

    # ALL_VC[0] = VC_B, ALL_VC[1] = VC_W, ALL_VC[2] = VC_C
    ALL_VC = np.array([VC_B, VC_W, VC_C])

    # ALL_FC[0] = FC_B, ALL_FC[1] = FC_W, ALL_FC[2] = FC_C
    ALL_FC = np.array([FC_B, FC_W, FC_C])

    # ALL_CAPA[0] = CAPA_B, ALL_CAPA[1] = CAPA_W, ALL_CAPA[2] = CAPA_C
    ALL_CAPA = np.stack([CAPA_B, CAPA_W, CAPA_C])

    ALL_AVA = np.stack([AVA_B, AVA_W, AVA_C])

    ORDER_ORDERTIMES = np.array(
        [o.order_time for o in all_orders]
    )

    ORDER_READYTIMES = np.array(
        [o.ready_time for o in all_orders]
    )

    ORDER_DEADLINES = np.array(
        [o.deadline for o in all_orders]
    )

    ORDER_VOLUMES = np.array(
        [o.volume for o in all_orders]
    )

    D = dist_mat

    def numpy_permutations(n):
        a = np.zeros((math.factorial(n), n), np.int16)
        f = 1
        for m in range(2, n + 1):
            b = a[:f, n - m + 1:]  # the block of permutations of range(m-1)
            for i in range(1, m):
                a[i * f:(i + 1) * f, n - m] = i
                a[i * f:(i + 1) * f, n - m + 1:] = b + (b >= i)
            b += 1
            f *= m

        return a

    MAX_BUNDLE_SIZE = 7

    # [[], [[0]], [[0 1]
    #  [1 0]], [[0 1 2]
    #  [0 2 1]
    #  [1 0 2]
    #  [1 2 0]
    #  [2 0 1] ...
    PRECALCULATED_PERMUTATIONS = numba.typed.List([
        numpy_permutations(n) for n in range(0, MAX_BUNDLE_SIZE + 1)
    ])

    numba.set_num_threads(4)

    candidate_1_orders = [np.array([o], dtype=np.int16) for o in range(K)]
    candidate_1_riders = [np.array([0, 1, 2], dtype=np.int16) for o in range(K)]

    # 주어진 order set에 가능한 riders 조함
    # [0] -> Bike only
    # [2] -> Car only
    # [0,2] -> Bike & Car
    # [0,1,2] -> All)
    # Walk만 가능한건 불가능!

    # feasible_riders_for_1_orders: 각 주문별로 배송 가능한 rider type
    ### 00 = {tuple: 2} ([0], [0 1 2])  0번 주문은 Bike, Walk, Car
    ### 04 = {tuple: 2} ([4], [0 2])    4번 주문은 Bike, Car

    # feasible_1_bundles, shop_seq, dlvry_seq, rider, dist, vol, cost, status
    ### 000 = {tuple: 6}(([0], [0]), 0, 1099, 18, 5659.4, 0)
    ### 001 = {tuple: 6}(([0], [0]), 1, 1099, 18, 5329.7, 0)
    ### 002 = {tuple: 6}(([0], [0]), 2, 1099, 18, 6099.0, 0)
    feasible_riders_for_1_orders, feasible_1_bundles = batch_get_optimal_route_with_riders_numba_v2(
        to_numba_List(candidate_1_orders), to_numba_List(candidate_1_riders), ORDER_READYTIMES, ORDER_DEADLINES,
        ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS, time_limit=10)
    print(
        f'* 1-bundle: candidate={sum([len(r) for r in feasible_riders_for_1_orders])}, feasible={len(feasible_1_bundles)}, elapsed time={time.time() - start_time}')
    all_feasible_bundles = feasible_1_bundles

    # candidate_2_orders: 2개씩 묶어볼만한 주문들
    # 0000 = {ndarray: (2,)} [0 1]
    # 0001 = {ndarray: (2,)} [0 2]
    # 0002 = {ndarray: (2,)} [0 3]
    # 0003 = {ndarray: (2,)} [0 4]
    # candidate_2_riders: 2개씩 묶었을 때 가능한 rider들
    # 0000 = {ndarray: (3,)} [0 1 2]
    # 0001 = {ndarray: (3,)} [0 1 2]
    # 0002 = {ndarray: (3,)} [0 1 2]
    # 0003 = {ndarray: (2,)} [0 2]
    # len(candidate_2_orders) = len(candidate_2_riders)
    candidate_2_orders, candidate_2_riders = make_2_bundle_candidates_v2(feasible_riders_for_1_orders)

    feasible_riders_for_2_orders, feasible_2_bundles = batch_get_optimal_route_with_riders_v2(candidate_2_orders,
                                                                                              candidate_2_riders,
                                                                                              ORDER_READYTIMES,
                                                                                              ORDER_DEADLINES,
                                                                                              ORDER_VOLUMES, D, ALL_T,
                                                                                              ALL_VC, ALL_FC, ALL_CAPA,
                                                                                              PRECALCULATED_PERMUTATIONS,
                                                                                              time_limit=time_limit2)

    print(
        f'* 2-bundle: candidate={sum([len(r) for r in candidate_2_riders])}, feasible={len(feasible_2_bundles)}, elapsed time={time.time() - start_time}')

    if (timelimit <= 15 and len(feasible_2_bundles) >= 200000) or (
            30 <= timelimit < 60 and len(feasible_2_bundles) >= 250000) or (
            60 <= timelimit < 70 and len(feasible_2_bundles) >= 500000):
        sorted_candidates = sorted(feasible_2_bundles, key=lambda bundle: bundle[4])
        feasible_2_bundles = sorted_candidates[:max_2bundles]
        feasible_riders_for_2_orders = [(bundle[0][0], np.array([bundle[1]], dtype=np.int16)) for bundle in
                                        feasible_2_bundles]
        all_feasible_bundles.extend(feasible_2_bundles)

    else:
        all_feasible_bundles.extend(feasible_2_bundles)

    elapsed_time = time.time() - start_time
    if (elapsed_time < timelimit * 0.5):

        # Penalty Matrix 만들기 (초기값은 100)
        penalty_mat = np.ones((K, K)) * 100
        for orders, _ in feasible_riders_for_2_orders:
            i, j = orders[0], orders[1]
            # i,j 간은 이동이 가능하기 때문에 cost가 0
            penalty_mat[i, j] = 0
            penalty_mat[j, i] = 0

        # 번들이 만들어질 가능성 점수 (값이 작을수록 번들 성립 가능성 높음)
        # 가게 거리 기준
        shop_dist_mat = D[:K, :K]
        shop_dist_bundle_score_mat = (shop_dist_mat) / shop_dist_mat.max()

        # 두개 번들이 불가능한 조합에 큰 페널티 값 더해줌
        shop_dist_bundle_score_mat += penalty_mat

        # i, j간 상대적인 거리를 계산
        lambda_value = 0.5  # shop과 dlvry 중 무엇에 가중치를 더 둘 것인가?
        weighted_dist_bundle_score_mat = calculate_weighted_dist_mat(all_orders, lambda_value)
        weighted_dist_bundle_score_mat += penalty_mat

        # 고객 거리 기준
        dlv_dist_mat = D[K:2 * K, K:2 * K]
        dlv_dist_bundle_score_mat = (dlv_dist_mat) / dlv_dist_mat.max()
        dlv_dist_bundle_score_mat += penalty_mat

        combined_bundle_score_mat = (
                shop_dist_bundle_score_mat +
                weighted_dist_bundle_score_mat +
                dlv_dist_bundle_score_mat
        )

        candidate_3_orders_shop, candidate_3_riders_shop = make_larger_bundle_candidates_v2(
            feasible_riders_for_2_orders, feasible_riders_for_1_orders, combined_bundle_score_mat, num_tries=index3)

        feasible_riders_for_3_orders, feasible_3_bundles = batch_get_optimal_route_with_riders_v2(
            candidate_3_orders_shop, candidate_3_riders_shop, ORDER_READYTIMES, ORDER_DEADLINES, ORDER_VOLUMES, D,
            ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS, time_limit=time_limit3)
        print(
            f'* 3-bundle: candidate={sum([len(r) for r in candidate_3_riders_shop])}, feasible={len(feasible_3_bundles)}, elapsed time={time.time() - start_time}')

        all_feasible_bundles.extend(feasible_3_bundles)

        candidate_4_orders_shop, candidate_4_riders_shop = make_larger_bundle_candidates_v2(
            feasible_riders_for_3_orders, feasible_riders_for_1_orders, combined_bundle_score_mat, num_tries=index4)

        feasible_riders_for_4_orders, feasible_4_bundles = batch_get_optimal_route_with_riders_v2(
            candidate_4_orders_shop, candidate_4_riders_shop, ORDER_READYTIMES, ORDER_DEADLINES, ORDER_VOLUMES, D,
            ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS, time_limit=time_limit4)
        print(
            f'* 4-bundle: candidate={sum([len(r) for r in candidate_4_riders_shop])}, feasible={len(feasible_4_bundles)}, elapsed time={time.time() - start_time}')

        all_feasible_bundles.extend(feasible_4_bundles)

        elapsed_time = time.time() - start_time
        if (len(feasible_4_bundles) >= 10) and (elapsed_time < timelimit * 0.3):
            candidate_5_orders_shop, candidate_5_riders_shop = make_larger_bundle_candidates_v2(
                feasible_riders_for_4_orders, feasible_riders_for_1_orders, combined_bundle_score_mat, num_tries=index5)

            feasible_riders_for_5_orders, feasible_5_bundles = batch_get_optimal_route_with_riders_v2(
                candidate_5_orders_shop, candidate_5_riders_shop, ORDER_READYTIMES, ORDER_DEADLINES, ORDER_VOLUMES, D,
                ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS, time_limit=time_limit5)
            print(
                f'* 5-bundle: candidate={sum([len(r) for r in candidate_5_riders_shop])}, feasible={len(feasible_5_bundles)}, elapsed time={time.time() - start_time}')

            all_feasible_bundles.extend(feasible_5_bundles)

            elapsed_time = time.time() - start_time
            if (timelimit > 15) and (len(feasible_5_bundles) >= 10) and (elapsed_time < timelimit * 0.3):
                candidate_6_orders_shop, candidate_6_riders_shop = make_larger_bundle_candidates_v2(
                    feasible_riders_for_5_orders, feasible_riders_for_1_orders, combined_bundle_score_mat,
                    num_tries=index6)

                if (len(candidate_6_orders_shop) > 0):
                    feasible_riders_for_6_orders, feasible_6_bundles = batch_get_optimal_route_with_riders_v2(
                        candidate_6_orders_shop, candidate_6_riders_shop, ORDER_READYTIMES, ORDER_DEADLINES,
                        ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS,
                        time_limit=time_limit6)
                    print(
                        f'* 6-bundle: candidate={sum([len(r) for r in candidate_6_riders_shop])}, feasible={len(feasible_6_bundles)}, elapsed time={time.time() - start_time}')

                    all_feasible_bundles.extend(feasible_6_bundles)

        print('all_feasible_bundles sample one ')
        # bundle_feasibility.append(
        #     ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status))
        print(all_feasible_bundles[200])

        if (timelimit >= 60):
            elapsed_time = time.time() - start_time
            if elapsed_time < timelimit * 0.3:
                print('* generating more bundles')

                index3 *= 2
                index4 *= 2
                index5 *= 2
                index6 *= 2

                # candidate_3_orders_shop, candidate_3_riders_shop = make_larger_bundle_candidates_v2(
                #     feasible_riders_for_2_orders, feasible_riders_for_1_orders, combined_bundle_score_mat,
                #     num_tries=index3)
                # feasible_riders_for_3_orders, feasible_3_bundles = batch_get_optimal_route_with_riders_v2(
                #     candidate_3_orders_shop, candidate_3_riders_shop, ORDER_READYTIMES, ORDER_DEADLINES, ORDER_VOLUMES,
                #     D,
                #     ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS, time_limit=time_limit3)

                candidate_3_orders_shop2, candidate_3_riders_shop2 = make_larger_bundle_candidates_v2(
                    feasible_riders_for_2_orders, feasible_riders_for_1_orders, combined_bundle_score_mat,
                    num_tries=index3)
                feasible_riders_for_3_orders2, feasible_3_bundles2 = batch_get_optimal_route_with_riders_v2(
                    candidate_3_orders_shop2, candidate_3_riders_shop2, ORDER_READYTIMES, ORDER_DEADLINES,
                    ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS,
                    time_limit=time_limit3)

                print(
                    f'* Extra 3-bundle: candidate={sum([len(r) for r in candidate_3_riders_shop2])}, feasible={len(feasible_3_bundles2)}, elapsed time={time.time() - start_time}')
                all_feasible_bundles.extend(feasible_3_bundles2)

                candidate_4_orders_shop2, candidate_4_riders_shop2 = make_larger_bundle_candidates_v2(
                    feasible_riders_for_3_orders2, feasible_riders_for_1_orders, combined_bundle_score_mat,
                    num_tries=index4)

                elapsed_time = time.time() - start_time
                if elapsed_time < timelimit * 0.3:
                    feasible_riders_for_4_orders2, feasible_4_bundles2 = batch_get_optimal_route_with_riders_v2(
                        candidate_4_orders_shop2, candidate_4_riders_shop2, ORDER_READYTIMES, ORDER_DEADLINES,
                        ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS,
                        time_limit=time_limit4)

                    print(
                        f'* Extra 4-bundle: candidate={sum([len(r) for r in candidate_4_riders_shop2])}, feasible={len(feasible_4_bundles2)}, elapsed time={time.time() - start_time}')
                    all_feasible_bundles.extend(feasible_4_bundles2)

                    elapsed_time = time.time() - start_time
                    if (len(feasible_4_bundles2) >= 10) and (elapsed_time < timelimit * 0.3):
                        candidate_5_orders_shop2, candidate_5_riders_shop2 = make_larger_bundle_candidates_v2(
                            feasible_riders_for_4_orders2, feasible_riders_for_1_orders, combined_bundle_score_mat,
                            num_tries=index5)

                        elapsed_time = time.time() - start_time
                        if elapsed_time < timelimit * 0.3:
                            feasible_riders_for_5_orders2, feasible_5_bundles2 = batch_get_optimal_route_with_riders_v2(
                                candidate_5_orders_shop2, candidate_5_riders_shop2, ORDER_READYTIMES, ORDER_DEADLINES,
                                ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS,
                                time_limit=time_limit5)

                            print(
                                f'* Extra 5-bundle: candidate={sum([len(r) for r in candidate_5_riders_shop2])}, feasible={len(feasible_5_bundles2)}, elapsed time={time.time() - start_time}')
                            all_feasible_bundles.extend(feasible_5_bundles2)

                            elapsed_time = time.time() - start_time
                            if (len(feasible_5_bundles2) >= 10) and (elapsed_time < timelimit * 0.3):

                                candidate_6_orders_shop2, candidate_6_riders_shop2 = make_larger_bundle_candidates_v2(
                                    feasible_riders_for_5_orders2, feasible_riders_for_1_orders,
                                    combined_bundle_score_mat, num_tries=index6)

                                elapsed_time = time.time() - start_time
                                if elapsed_time < timelimit * 0.3:
                                    feasible_riders_for_6_orders2, feasible_6_bundles2 = batch_get_optimal_route_with_riders_v2(
                                        candidate_6_orders_shop2, candidate_6_riders_shop2, ORDER_READYTIMES,
                                        ORDER_DEADLINES, ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA,
                                        PRECALCULATED_PERMUTATIONS, time_limit=time_limit6)

                                    print(
                                        f'* Extra 6-bundle: candidate={sum([len(r) for r in candidate_6_riders_shop2])}, feasible={len(feasible_6_bundles2)}, elapsed time={time.time() - start_time}')
                                    all_feasible_bundles.extend(feasible_6_bundles2)

                print(
                    f'* 중복 제거 전 all-bundle: feasible={len(all_feasible_bundles)}, elapsed time={time.time() - start_time}')
                # (1,2,3) , (2,3,1), (3,1,2) 모두 중복이다
                _, all_feasible_bundles = remove_duplicates_in_bundles([], all_feasible_bundles)

    print(f'* 중복 제거 후 all-bundle: feasible={len(all_feasible_bundles)}, elapsed time={time.time() - start_time}')

    remaining_time = timelimit - (time.time() - start_time)

    file_name = 'all_feasible_bundles_K50_1.txt'
    # bundle_feasibility.append(
    #     ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status))
    # delimiter = '///'
    # for feasible_bundles in all_feasible_bundles:
    #     shop_seq = feasible_bundles[0][0]
    #     dlvry_seq = feasible_bundles[0][1]
    #     print(shop_seq)
    #     rider_type = feasible_bundles[1]
    #     dist = feasible_bundles[2]
    #     vol = feasible_bundles[3]
    #     cost = feasible_bundles[4]
    #     status = feasible_bundles[5]
    #
    #     print(
    #         shop_seq + delimiter + dlvry_seq + delimiter + rider_type + delimiter + rider_type + delimiter + dist + delimiter + vol + delimiter + cost + delimiter + status)

    final_bundles = solve_mip(all_feasible_bundles, ALL_AVA, K, remaining_time - slack, covering=True)

    final_bundles, status = find_cross_bundle_duplicates_and_generate_subsets(final_bundles, ORDER_READYTIMES,
                                                                              ORDER_DEADLINES, ORDER_VOLUMES, D, ALL_T,
                                                                              ALL_VC, ALL_FC, ALL_CAPA,
                                                                              PRECALCULATED_PERMUTATIONS)

    if status == True:
        remaining_time = timelimit - (time.time() - start_time)
        final_bundles = solve_mip(final_bundles, ALL_AVA, K, remaining_time - 0.5, covering=False)

    RIDER_TYPE = {0: 'BIKE', 1: 'WALK', 2: 'CAR'}
    solution = [
        [RIDER_TYPE[bundle[1]], bundle[0][0].tolist(), bundle[0][1].tolist()]
        for bundle in final_bundles
    ]

    return solution


def get_conditions():
    conditions = [

        # 1016
        {
            "condition": lambda K, timelimit: timelimit < 30 and K < 201,
            "params": {"index3": 4, "index4": 4, "index5": 3, "index6": 3,
                       "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 3, "time_limit6": 3,
                       "max_2bundles": 80000, "slack": 3}
        },
        {
            "condition": lambda K, timelimit: timelimit < 30 and 201 <= K < 301,
            "params": {"index3": 3, "index4": 3, "index5": 2, "index6": 2,
                       "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 3, "time_limit6": 3,
                       "max_2bundles": 80000, "slack": 3}
        },
        {
            "condition": lambda K, timelimit: timelimit < 30 and 301 <= K < 501,
            "params": {"index3": 1, "index4": 2, "index5": 2, "index6": 1,
                       "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 1, "time_limit6": 1,
                       "max_2bundles": 80000, "slack": 3}
        },
        {
            "condition": lambda K, timelimit: timelimit < 30 and 501 <= K < 1001,
            "params": {"index3": 1, "index4": 1, "index5": 2, "index6": 1,
                       "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 2, "time_limit6": 1,
                       "max_2bundles": 80000, "slack": 3}
        },
        {
            "condition": lambda K, timelimit: timelimit < 30 and 1001 <= K < 2001,
            "params": {"index3": 1, "index4": 1, "index5": 1, "index6": 1,
                       "time_limit2": 1, "time_limit3": 1, "time_limit4": 1, "time_limit5": 1, "time_limit6": 1,
                       "max_2bundles": 60000, "slack": 3}
        },

        {
            "condition": lambda K, timelimit: 30 <= timelimit < 60 and K < 201,
            "params": {"index3": 4, "index4": 3, "index5": 3, "index6": 3,
                       "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 3, "time_limit6": 3,
                       "max_2bundles": 150000, "slack": 4}
        },
        {
            "condition": lambda K, timelimit: 30 <= timelimit < 60 and 201 <= K < 301,
            "params": {"index3": 3, "index4": 2, "index5": 2, "index6": 2,
                       "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 3, "time_limit6": 3,
                       "max_2bundles": 150000, "slack": 4}
        },
        {
            "condition": lambda K, timelimit: 30 <= timelimit < 60 and 301 <= K < 501,
            "params": {"index3": 1, "index4": 1, "index5": 2, "index6": 2,
                       "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 3, "time_limit6": 3,
                       "max_2bundles": 150000, "slack": 4}
        },
        {
            "condition": lambda K, timelimit: 30 <= timelimit < 60 and 501 <= K < 1001,
            "params": {"index3": 1, "index4": 1, "index5": 1, "index6": 1,
                       "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 2, "time_limit6": 3,
                       "max_2bundles": 150000, "slack": 4}
        },
        {
            "condition": lambda K, timelimit: 30 <= timelimit < 60 and 1001 <= K < 2001,
            "params": {"index3": 1, "index4": 1, "index5": 1, "index6": 1,
                       "time_limit2": 1, "time_limit3": 1, "time_limit4": 1, "time_limit5": 1, "time_limit6": 1,
                       "max_2bundles": 80000, "slack": 4}
        },

        # 1013
        {
            "condition": lambda K, timelimit: 60 <= timelimit < 70 and K < 301,
            "params": {"index3": 3, "index4": 3, "index5": 3, "index6": 3,
                       "time_limit2": 10, "time_limit3": 5, "time_limit4": 5, "time_limit5": 5, "time_limit6": 5,
                       "max_2bundles": 200000, "slack": 5}
        },
        {
            "condition": lambda K, timelimit: 60 <= timelimit < 70 and 301 <= K < 501,
            "params": {"index3": 2, "index4": 2, "index5": 1, "index6": 1,
                       "time_limit2": 10, "time_limit3": 3, "time_limit4": 3, "time_limit5": 8, "time_limit6": 8,
                       "max_2bundles": 200000, "slack": 5}
        },
        {
            "condition": lambda K, timelimit: 60 <= timelimit < 70 and 501 <= K < 1001,
            "params": {"index3": 2, "index4": 1, "index5": 1, "index6": 1,
                       "time_limit2": 10, "time_limit3": 3, "time_limit4": 3, "time_limit5": 5, "time_limit6": 5,
                       "max_2bundles": 80000, "slack": 5}
        },
        {
            "condition": lambda K, timelimit: 60 <= timelimit < 70 and 1001 <= K < 2001,
            "params": {"index3": 1, "index4": 1, "index5": 1, "index6": 1,
                       "time_limit2": 1, "time_limit3": 2, "time_limit4": 2, "time_limit5": 2, "time_limit6": 2,
                       "max_2bundles": 100000, "slack": 5}
        },

        {
            "condition": lambda K, timelimit: 70 <= timelimit < 150 and K < 301,
            "params": {"index3": 3, "index4": 3, "index5": 3, "index6": 3,
                       "time_limit2": 10, "time_limit3": 5, "time_limit4": 5, "time_limit5": 10, "time_limit6": 15,
                       "slack": 12}
        },
        {
            "condition": lambda K, timelimit: 70 <= timelimit < 150 and 301 <= K < 501,
            "params": {"index3": 2, "index4": 2, "index5": 2, "index6": 3,
                       "time_limit2": 10, "time_limit3": 5, "time_limit4": 5, "time_limit5": 10, "time_limit6": 15,
                       "slack": 12}
        },
        {
            "condition": lambda K, timelimit: 70 <= timelimit < 150 and 501 <= K < 1001,
            "params": {"index3": 2, "index4": 2, "index5": 2, "index6": 2,
                       "time_limit2": 10, "time_limit3": 2, "time_limit4": 2, "time_limit5": 3, "time_limit6": 3,
                       "slack": 12}
        },
        {
            "condition": lambda K, timelimit: 70 <= timelimit < 150 and 1001 <= K < 2001,
            "params": {"index3": 1, "index4": 1, "index5": 1, "index6": 1,
                       "time_limit2": 10, "time_limit3": 3, "time_limit4": 3, "time_limit5": 5, "time_limit6": 5,
                       "slack": 12}
        },

        # 1013
        {
            "condition": lambda K, timelimit: 150 <= timelimit < 290 and K < 500,
            "params": {"index3": 3, "index4": 3, "index5": 3, "index6": 2,
                       "time_limit2": 10, "time_limit3": 10, "time_limit4": 10, "time_limit5": 30, "time_limit6": 50,
                       "slack": 18}
        },
        {
            "condition": lambda K, timelimit: 150 <= timelimit < 290 and 500 <= K < 749,
            "params": {"index3": 3, "index4": 3, "index5": 2, "index6": 2,
                       "time_limit2": 10, "time_limit3": 10, "time_limit4": 10, "time_limit5": 30, "time_limit6": 50,
                       "slack": 18}
        },
        ###
        {
            "condition": lambda K, timelimit: 150 <= timelimit < 290 and 749 <= K < 1001,
            "params": {"index3": 2, "index4": 2, "index5": 3, "index6": 3,
                       "time_limit2": 10, "time_limit3": 10, "time_limit4": 10, "time_limit5": 30, "time_limit6": 30,
                       "slack": 18}
        },
        {
            "condition": lambda K, timelimit: 150 <= timelimit < 290 and 1001 <= K < 2001,
            "params": {"index3": 1, "index4": 1, "index5": 1, "index6": 1,
                       "time_limit2": 10, "time_limit3": 5, "time_limit4": 5, "time_limit5": 30, "time_limit6": 30,
                       "slack": 18}
        },

        {
            "condition": lambda K, timelimit: 290 <= timelimit < 350 and K < 501,
            "params": {"index3": 4, "index4": 4, "index5": 5, "index6": 5,
                       "time_limit2": 10, "time_limit3": 10, "time_limit4": 30, "time_limit5": 60, "time_limit6": 80,
                       "slack": 25}
        },
        {
            "condition": lambda K, timelimit: 290 <= timelimit < 350 and 501 <= K < 1001,
            "params": {"index3": 4, "index4": 4, "index5": 4, "index6": 4,
                       "time_limit2": 10, "time_limit3": 10, "time_limit4": 30, "time_limit5": 60, "time_limit6": 80,
                       "slack": 25}
        },
        {
            "condition": lambda K, timelimit: 290 <= timelimit < 350 and 1001 <= K < 2001,
            "params": {"index3": 2, "index4": 1, "index5": 1, "index6": 1,
                       "time_limit2": 10, "time_limit3": 10, "time_limit4": 10, "time_limit5": 40, "time_limit6": 40,
                       "slack": 25}
        },

        {
            "condition": lambda K, timelimit: 350 <= timelimit < 500 and K < 501,
            "params": {"index3": 4, "index4": 5, "index5": 5, "index6": 5,
                       "time_limit2": 10, "time_limit3": 10, "time_limit4": 30, "time_limit5": 60, "time_limit6": 120,
                       "slack": 20}
        },
        {
            "condition": lambda K, timelimit: 350 <= timelimit < 500 and 501 <= K < 1001,
            "params": {"index3": 4, "index4": 4, "index5": 4, "index6": 4,
                       "time_limit2": 10, "time_limit3": 10, "time_limit4": 30, "time_limit5": 80, "time_limit6": 120,
                       "slack": 20}
        },
        {
            "condition": lambda K, timelimit: 350 <= timelimit < 500 and 1001 <= K < 2001,
            "params": {"index3": 2, "index4": 1, "index5": 1, "index6": 1,
                       "time_limit2": 10, "time_limit3": 10, "time_limit4": 10, "time_limit5": 40, "time_limit6": 40,
                       "slack": 20}
        }
    ]
    return conditions


# Function to convert the list to the desired format and save to a text file
# def save_list_to_text_file(data, filename):
#     with open(filename, 'w') as file:
#         # for item in data:
#         #     if isinstance(item, list):
#         #         file.write(f"[{' '.join(map(str, item))}], ")
#         #     else:
#         #         file.write(f"{item}, ")
#         # file.write("\n")
#         # Format the first two sublists
#         formatted_str = f"([{ ' '.join(map(str, data[0][0])) }], [{ ' '.join(map(str, data[0][1])) }]), " # Add the remaining elements
#         formatted_str += ', '.join(map(str, data[1:])) + '\n'
#         file.write(formatted_str)

def solve_mip(feasible_bundles, ALL_AVA, K, timelimit, covering=False):
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

    # elif m.status == GRB.INFEASIBLE:
    #     print('Model is infeasible')
    #     m.computeIIS()
    #     m.write("infeasible_model.ilp")

    else:
        print('No solution found.')

    return final_bundles


def find_cross_bundle_duplicates_and_generate_subsets(final_bundles, ORDER_READYTIMES, ORDER_DEADLINES, ORDER_VOLUMES,
                                                      D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS):
    order_to_bundles = {}

    for idx, bundle in enumerate(final_bundles):
        shop_seq, dlv_seq = bundle[0]
        shop_orders = tuple(sorted(shop_seq))

        # order별로 들어있는 bundle idx 
        for order in set(shop_orders):
            if order not in order_to_bundles:
                order_to_bundles[order] = []
            order_to_bundles[order].append(idx)

    # 중복이 발생하는 bundle idx
    bundles_with_duplicates = set()
    for order, indices in order_to_bundles.items():
        if len(indices) > 1:
            bundles_with_duplicates.update(indices)

    status = False

    if not bundles_with_duplicates:
        print('** 중복 없음 **')
        return final_bundles, status

    else:
        print('** 중복 발생! **')
        status = True

    duplicate_bundles = [final_bundles[i] for i in sorted(bundles_with_duplicates)]

    subset_bundles = []
    for bundle in duplicate_bundles:
        shop_seq, dlv_seq = bundle[0]

        for r in range(1, len(shop_seq) + 1):
            shop_subsets = combinations(shop_seq, r)
            for shop_subset in shop_subsets:
                subset_bundles.append(np.array(shop_subset, dtype=np.int16))

    feasible_riders_for_subset, feasible_subset_bundles = batch_get_optimal_route_with_riders_numba_v2(
        to_numba_List(subset_bundles), to_numba_List([np.array([0, 1, 2], dtype=np.int16)] * len(subset_bundles)),
        ORDER_READYTIMES, ORDER_DEADLINES, ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA,
        PRECALCULATED_PERMUTATIONS, 3)

    final_bundles += feasible_subset_bundles

    return final_bundles, status


# @numba.njit(cache=True, parallel=True)
def batch_get_optimal_route_numba_v2(set_orders, set_riders, ORDER_READYTIMES, ORDER_DEADLINES, ORDER_VOLUMES, D, ALL_T,
                                     ALL_VC, ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS):
    feasible_bundles = [((np.array([1], dtype=np.int16), np.array([1], dtype=np.int16)), -1, -1, -1, -1.0)] * len(
        set_orders)
    for idx in numba.prange(len(set_orders)):
        orders, rider = set_orders[idx], set_riders[idx]
        status, dist, (shop_seq, dlv_seq), vol, cost = get_optimal_route_numba_v2(orders, rider, ORDER_READYTIMES,
                                                                                  ORDER_DEADLINES, ORDER_VOLUMES, D,
                                                                                  ALL_T, ALL_VC, ALL_FC, ALL_CAPA,
                                                                                  PRECALCULATED_PERMUTATIONS)
        if status == 0:  # feasible
            feasible_bundles[idx] = ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost)

    return [fb for fb in feasible_bundles if fb[-1] > 0]


# @numba.njit(cache=True, fastmath=True, nogil=True)
def get_optimal_route_numba_v2(orders, rider, ORDER_READYTIMES, ORDER_DEADLINES, ORDER_VOLUMES, D, ALL_T, ALL_VC,
                               ALL_FC, ALL_CAPA, PRECALCULATED_PERMUTATIONS):
    K = len(ORDER_READYTIMES)

    def get_total_volume(shop_seq, ORDER_VOLUMES):
        total_vol = 0
        for o in shop_seq:
            total_vol += ORDER_VOLUMES[o]
        return total_vol

    def calculate_cost(total_dist):
        return ALL_FC[rider] + (total_dist / 100.0) * ALL_VC[rider]

    orders = orders.astype(np.int16)

    # total volume check
    total_vol = sum([ORDER_VOLUMES[o] for o in orders])
    if total_vol > ALL_CAPA[rider]:
        return -1, 0, (orders, orders), 0, 0.0

    # latest shop ready time + lead time > earlist delivery time
    latest_readytime_order = orders[np.argmax(ORDER_READYTIMES[orders])]
    earlest_deadline_order = orders[np.argmin(ORDER_DEADLINES[orders])]
    if ALL_T[rider][latest_readytime_order, earlest_deadline_order + K] + ORDER_READYTIMES[latest_readytime_order] > \
            ORDER_DEADLINES[earlest_deadline_order]:
        return -3, 0, (orders, orders), 0, 0.0

    # if len(orders) > 1:
    #     t = 1
    # n = 2
    # index_permutations : numpy array [[0,1], [1,0]]
    # order_permutations : list [[0,1], [1,0]]
    n = len(orders)
    index_permutations = PRECALCULATED_PERMUTATIONS[n]
    order_permutations = []
    for i in range(len(index_permutations)):
        order_permutations.append(orders[index_permutations[i]])

    # 각 order permutation 에 해당하는 customer delivery에 소요되는 distance, arrival time 계산
    dlv_distance_lookuptable = []
    dlv_arrivaltimes_lookuptable = []

    for order_seq in order_permutations:
        dlv_dist = 0
        dlv_time = 0
        dlv_times = [dlv_time]

        for i, j in zip(order_seq[:-1], order_seq[1:]):
            dlv_dist += D[i + K, j + K]
            dlv_time += ALL_T[rider][i + K, j + K]
            dlv_times.append(dlv_time)

        dlv_distance_lookuptable.append(dlv_dist)
        dlv_arrivaltimes_lookuptable.append(np.array(dlv_times))

    min_dist = 100000000
    opt_shop_seq = orders
    opt_dlv_seq = orders
    feasible = False

    for shop_seq in order_permutations:
        rt = ORDER_READYTIMES[shop_seq[0]]
        shop_dist = 0
        for i, j in zip(shop_seq[:-1], shop_seq[1:]):
            rt = max(rt + ALL_T[rider][i, j], ORDER_READYTIMES[j])
            shop_dist += D[i, j]

        # max ready time + moving time from last shop to first order > first order deadline
        if rt + ALL_T[rider][shop_seq[-1], earlest_deadline_order + K] > ORDER_DEADLINES[earlest_deadline_order]:
            continue

        for dlv_perm_idx, dlv_seq in enumerate(order_permutations):
            feasibility_check = True

            dlv_st = rt + ALL_T[rider][shop_seq[-1], dlv_seq[0] + K]
            dlv_times = dlv_arrivaltimes_lookuptable[dlv_perm_idx]
            for dlv_time, k in zip(dlv_times, dlv_seq):
                if dlv_st + dlv_time > ORDER_DEADLINES[k]:
                    feasibility_check = False
                    break

            if feasibility_check:
                feasible = True
                total_dist = shop_dist + D[shop_seq[-1], dlv_seq[0] + K] + dlv_distance_lookuptable[dlv_perm_idx]
                total_vol = get_total_volume(shop_seq, ORDER_VOLUMES)

                # print(f'{total_dist=}, {shop_seq=}, {dlv_seq=} ')
                if total_dist < min_dist:
                    min_dist = total_dist
                    opt_shop_seq = shop_seq
                    opt_dlv_seq = dlv_seq

    if not feasible:
        return -2, 0, (orders, orders), 0, 0.0

    return 0, min_dist, (opt_shop_seq, opt_dlv_seq), total_vol, calculate_cost(min_dist)


# @numba.njit(cache=True)
def get_total_distance_numba(K, dist_mat, shop_seq, dlv_seq):
    if len(shop_seq) == 0:
        return 0

    if len(shop_seq) == 1:
        return dist_mat[shop_seq[0], shop_seq[0] + K]

    dist = 0
    for i, j in zip(shop_seq[:-1], shop_seq[1:]):
        dist += dist_mat[i, j]

    dist += dist_mat[shop_seq[-1], dlv_seq[0] + K]

    for i, j in zip(dlv_seq[:-1], dlv_seq[1:]):
        dist += dist_mat[i + K, j + K]

    return dist


# convex combination
def calculate_weighted_position(order, lambda_value):
    shop_lat, shop_lon = order.shop_lat, order.shop_lon
    delivery_lat, delivery_lon = order.dlv_lat, order.dlv_lon
    weighted_lat = (1 - lambda_value) * shop_lat + lambda_value * delivery_lat
    weighted_lon = (1 - lambda_value) * shop_lon + lambda_value * delivery_lon
    return weighted_lat, weighted_lon


def calculate_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)  # 두 좌표 간의 유클리드 거리


def calculate_weighted_dist_mat(all_orders, lambda_value=None):
    num_orders = len(all_orders)
    weighted_dist_mat = np.zeros((num_orders, num_orders))

    for i, order1 in enumerate(all_orders):
        lat1, lon1 = calculate_weighted_position(order1, lambda_value)
        for j, order2 in enumerate(all_orders):
            if i != j:
                lat2, lon2 = calculate_weighted_position(order2, lambda_value)
                weighted_dist_mat[i, j] = calculate_distance(lat1, lon1, lat2, lon2)

    return weighted_dist_mat


def to_numba_List(python_list):
    numba_List = numba.typed.List()
    for l in python_list:
        numba_List.append(l)
    return numba_List


# candidate 2에 있는 것들 중 1에 없는 것들만 반환
def get_differences_in_candidates(candidate_orders1, candidate_orders2, candidate_riders2):
    candidate_orders2_diff = []
    candidate_riders2_diff = []

    candidate_orders1_set = set([tuple(co) for co in candidate_orders1])

    for co, cr in zip(candidate_orders2, candidate_riders2):
        co_tuple = tuple(co)
        if co_tuple not in candidate_orders1_set:
            candidate_orders2_diff.append(co)
            candidate_riders2_diff.append(cr)

    return candidate_orders2_diff, candidate_riders2_diff


def remove_duplicates_in_candidates(candidate_orders, candidate_riders):
    candidate_orders_set = set()
    candidate_orders_no_dup = []
    candidate_riders_no_dup = []

    for co, cr in zip(candidate_orders, candidate_riders):
        co_tuple = tuple(co)
        if co_tuple not in candidate_orders_set:
            candidate_orders_no_dup.append(co)
            candidate_riders_no_dup.append(cr)
            candidate_orders_set.add(co_tuple)

    return candidate_orders_no_dup, candidate_riders_no_dup


def remove_duplicates_in_bundles(feasible_riders_for_orders, feasible_bundles):
    # feasible_riders_for_orders_dict = {
    #     tuple(np.sort(ro[0])): ro[1] for ro in feasible_riders_for_orders
    # }
    # feasible_riders_for_orders_no_dup = [
    #     (np.array(k, dtype=np.int16), v) for k, v in feasible_riders_for_orders_dict.items()
    # ]
    feasible_bundles_no_dup = []
    feasible_bundles_set = set()
    for bd in feasible_bundles:
        tuple_bd = (tuple(np.sort(bd[0][0])), bd[1])  # order sequence, rider type
        if tuple_bd not in feasible_bundles_set:
            feasible_bundles_set.add(tuple_bd)
            feasible_bundles_no_dup.append(bd)
        else:
            print(f'removed ${tuple_bd}')
            t = 1

    return [], feasible_bundles_no_dup


# @numba.njit(cache=True)
def get_cur_time_numba(get_system_clock, as_seconds_double):
    system_clock = get_system_clock()
    current_time = as_seconds_double(system_clock)
    return current_time


# @numba.njit('Array(i2,1,"C")(Array(i2,1,"C"), Array(i2,1,"C"))', cache=True)
def get_rider_intersaction(riders1, riders2):
    return np.array([r for r in [0, 1, 2] if r in riders1 and r in riders2], dtype=np.int16)


# 두개 묶음 candidate 번들을 만듬
# @numba.njit('(List(UniTuple(Array(i2, 1, "C"),2),True),)', cache=True, fastmath=True, parallel=False, nogil=True)
def make_2_bundle_candidates_v2(feasible_riders_for_1_orders):
    candidate_2_orders = []
    candidate_2_riders = []

    n = len(feasible_riders_for_1_orders)

    for idx_1 in range(n - 1):
        orders_1, riders_1 = feasible_riders_for_1_orders[idx_1]
        i = orders_1[0]
        for idx_2 in range(idx_1 + 1, n):
            orders_2, riders_2 = feasible_riders_for_1_orders[idx_2]
            j = orders_2[0]
            riders = get_rider_intersaction(riders_1, riders_2)
            if len(riders) > 0:
                candidate_2_orders.append(np.array([i, j], dtype=np.int16))
                candidate_2_riders.append(riders)

    return candidate_2_orders, candidate_2_riders


# 주어진 feasible_riders_for_orders에서 주문을 하나 더 묶는 candidate 번들을 최대 num_tries개 만듬
def make_larger_bundle_candidates_v2(feasible_riders_for_orders, feasible_riders_for_1_orders, bundle_score_mat,
                                     num_tries=3, timelimit=100):
    starttime = time.time()

    candidate_more_orders = []
    candidate_more_riders = []

    candidate_more_orders_set = set()

    num_cpus = 4

    feasible_riders_for_orders = to_numba_List(feasible_riders_for_orders)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = []
        size_one_thread = math.ceil(len(feasible_riders_for_orders) // num_cpus)

        for i in range(num_cpus):
            start_idx = size_one_thread * i
            end_idx = size_one_thread * (i + 1)

            # print(start_idx, end_idx, timelimit-(time.time()-starttime))

            futures.append(
                executor.submit(larger_bundle_candidates_numba_v2, feasible_riders_for_orders[start_idx:end_idx],
                                feasible_riders_for_1_orders, bundle_score_mat, num_tries,
                                max(timelimit - (time.time() - starttime), 1))
            )

        for future in concurrent.futures.as_completed(futures):
            r1, r2 = future.result()

            # print(len(r1))

            for orders, riders in zip(r1, r2):
                orders_tuple = tuple(orders)
                if orders_tuple not in candidate_more_orders_set:
                    candidate_more_orders_set.add(orders_tuple)

                    candidate_more_orders.append(orders)
                    candidate_more_riders.append(riders)

    return candidate_more_orders, candidate_more_riders


# @numba.njit('(ListType(UniTuple(Array(i2, 1, "C"), 2)),List(UniTuple(Array(i2, 1, "C"),2), True),Array(f8, 2, "C"),i8,f8)', cache=True, fastmath=True, parallel=False, nogil=True)
def larger_bundle_candidates_numba_v2(feasible_riders_for_orders, feasible_riders_for_1_orders, bundle_score_mat,
                                      num_tries, timelimit):
    with numba.objmode(starttime='f8'):
        starttime = time.time()

    candidate_more_orders = []
    candidate_more_riders = []

    for idx, (orders, riders) in enumerate(feasible_riders_for_orders):

        if (idx % 1000) == 0:

            with numba.objmode(curtime='f8'):
                curtime = time.time()

            # print(curtime - starttime, timelimit)
            if curtime - starttime > timelimit:
                print('timelimit!!!')
                break

        # 점수 합 벡터
        total_scores = bundle_score_mat[orders].sum(axis=0)

        if num_tries == 1:
            args = np.array([total_scores.argmin()])
        else:
            args = np.argpartition(total_scores, num_tries)[:num_tries]

        for i in args:
            if total_scores[i] < 100:

                orders_tuple = sorted(list(orders) + [i])
                # 새로운 번들에 사용가능한 rider의 집합을 구함
                rider_candidates_for_merge = feasible_riders_for_1_orders[i][1]
                new_riders = get_rider_intersaction(riders, rider_candidates_for_merge)
                if len(new_riders) > 0:
                    candidate_more_orders.append(np.array(orders_tuple, dtype=np.int16))
                    candidate_more_riders.append(new_riders)

    return candidate_more_orders, candidate_more_riders


def batch_get_optimal_route_with_riders_v2(candidate_orders, candidate_riders, ORDER_READYTIMES, ORDER_DEADLINES,
                                           ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA,
                                           PRECALCULATED_PERMUTATIONS, time_limit=10.0):
    def get_optimal_route(candidate_orders, candidate_riders, time_limit):

        starttime = time.time()

        if isinstance(candidate_orders, list):
            candidate_orders = to_numba_List(candidate_orders)

        if isinstance(candidate_riders, list):
            candidate_riders = to_numba_List(candidate_riders)

        return batch_get_optimal_route_with_riders_numba_v2(candidate_orders, candidate_riders, ORDER_READYTIMES,
                                                            ORDER_DEADLINES, ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC,
                                                            ALL_CAPA, PRECALCULATED_PERMUTATIONS,
                                                            time_limit - (time.time() - starttime))

    num_cpus = 4

    feasible_riders_for_orders = []
    bundle_feasibility = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = []
        size_one_thread = math.ceil(len(candidate_orders) // num_cpus)

        for i in range(num_cpus):
            start_idx = size_one_thread * i
            end_idx = size_one_thread * (i + 1)
            futures.append(
                executor.submit(get_optimal_route, candidate_orders[start_idx:end_idx],
                                candidate_riders[start_idx:end_idx], time_limit=time_limit)
            )

        for future in concurrent.futures.as_completed(futures):
            r1, r2 = future.result()
            feasible_riders_for_orders.extend(r1)
            bundle_feasibility.extend(r2)

    return feasible_riders_for_orders, bundle_feasibility


# @numba.njit('(ListType(Array(i2,1,"C")), ListType(Array(i2,1,"C")), Array(i8,1,"C"), Array(i8,1,"C"), Array(i8,1,"C"), Array(i8,2,"C"), Array(f8,3,"C"), Array(i8,1,"C"), Array(i8,1,"C"), Array(i8,1,"C"), ListType(Array(i2,2,"C")), i8)', cache=True, parallel=False, nogil=True, fastmath=True)
def batch_get_optimal_route_with_riders_numba_v2(candidate_orders, candidate_riders, ORDER_READYTIMES, ORDER_DEADLINES,
                                                 ORDER_VOLUMES, D, ALL_T, ALL_VC, ALL_FC, ALL_CAPA,
                                                 PRECALCULATED_PERMUTATIONS, time_limit=10.0):
    # bundle_feasibility = [((np.array([1], dtype=np.int16), np.array([1], dtype=np.int16)), -1, -1, -1, -1.0, -1)] * len(candidate_orders) * 3
    # feasible_riders_for_orders = [(np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.int16))] * len(candidate_orders)

    # bundle_feasibility = numba.typed.List.empty_list('Tuple(UniTuple(Array(i2, 1, "C"), 2), int64, int64, int64, float64, int64)')
    # feasible_riders_for_orders = numba.typed.List.empty_list('Tuple(UniTuple(Array(i2, 1, "C"), 2)')

    bundle_feasibility = []
    feasible_riders_for_orders = []

    # starttime = get_cur_time_numba(get_system_clock, as_seconds_double)
    with numba.objmode(starttime='f8'):
        starttime = time.time()

    terminate = False

    for idx in range(len(candidate_orders)):

        if time_limit > 0 and (idx % 1000) == 0:
            # curtime = get_cur_time_numba(get_system_clock, as_seconds_double)
            with numba.objmode(curtime='f8'):
                curtime = time.time()

            # print(curtime - starttime, timelimit)
            if (curtime - starttime > time_limit):
                terminate = True
                print('Timelimit!!!')
                break

        orders = candidate_orders[idx]
        riders = candidate_riders[idx]

        fb_idx = idx * 3

        feasible_riders = []

        if 0 in riders:
            rider = 0  # Bike

            status, dist, (shop_seq, dlv_seq), vol, cost = get_optimal_route_numba_v2(orders, rider, ORDER_READYTIMES,
                                                                                      ORDER_DEADLINES, ORDER_VOLUMES, D,
                                                                                      ALL_T, ALL_VC, ALL_FC, ALL_CAPA,
                                                                                      PRECALCULATED_PERMUTATIONS)

            if status == 0:  # feasible
                # bundle_feasibility[fb_idx + rider] = ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status)
                bundle_feasibility.append(
                    ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status))
                feasible_riders.append(rider)

                if 1 in riders:
                    rider = 1
                    status, dist, (shop_seq, dlv_seq), vol, cost = get_optimal_route_numba_v2(orders, rider,
                                                                                              ORDER_READYTIMES,
                                                                                              ORDER_DEADLINES,
                                                                                              ORDER_VOLUMES, D, ALL_T,
                                                                                              ALL_VC, ALL_FC, ALL_CAPA,
                                                                                              PRECALCULATED_PERMUTATIONS)

                    if status == 0:
                        # bundle_feasibility[fb_idx + rider] = ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status)
                        bundle_feasibility.append(
                            ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status))
                        feasible_riders.append(rider)

                if 2 in riders:
                    rider = 2
                    status, dist, (shop_seq, dlv_seq), vol, cost = get_optimal_route_numba_v2(orders, rider,
                                                                                              ORDER_READYTIMES,
                                                                                              ORDER_DEADLINES,
                                                                                              ORDER_VOLUMES, D, ALL_T,
                                                                                              ALL_VC, ALL_FC, ALL_CAPA,
                                                                                              PRECALCULATED_PERMUTATIONS)
                    if status == 0:
                        # bundle_feasibility[fb_idx + rider] = ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status)
                        bundle_feasibility.append(
                            ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status))
                        feasible_riders.append(rider)

            else:
                if status == -1:
                    rider = 2  # Car
                    status, dist, (shop_seq, dlv_seq), vol, cost = get_optimal_route_numba_v2(orders, rider,
                                                                                              ORDER_READYTIMES,
                                                                                              ORDER_DEADLINES,
                                                                                              ORDER_VOLUMES, D, ALL_T,
                                                                                              ALL_VC, ALL_FC, ALL_CAPA,
                                                                                              PRECALCULATED_PERMUTATIONS)
                    if status == 0:
                        # bundle_feasibility[fb_idx + rider] = ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status)
                        bundle_feasibility.append(
                            ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status))
                        feasible_riders.append(rider)

        else:
            if 2 in riders:
                rider = 2
                status, dist, (shop_seq, dlv_seq), vol, cost = get_optimal_route_numba_v2(orders, rider,
                                                                                          ORDER_READYTIMES,
                                                                                          ORDER_DEADLINES,
                                                                                          ORDER_VOLUMES, D, ALL_T,
                                                                                          ALL_VC, ALL_FC, ALL_CAPA,
                                                                                          PRECALCULATED_PERMUTATIONS)
                if status == 0:
                    # bundle_feasibility[fb_idx + rider] = ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status)
                    bundle_feasibility.append(
                        ((shop_seq.astype(np.int16), dlv_seq.astype(np.int16)), rider, dist, vol, cost, status))
                    feasible_riders.append(rider)

        if len(feasible_riders) > 0:
            feasible_riders_for_orders.append((orders.astype(np.int16), np.array(feasible_riders, dtype=np.int16)))

    # return [rb for rb in feasible_riders_for_orders if len(rb[1]) > 0], [fb for fb in bundle_feasibility if fb[-1] == 0]
    return feasible_riders_for_orders, bundle_feasibility
