from util import *
import sys

'''
수정 사항 (by sehwa)
line 79에 print(e) --> raise Exception(str(e)) by sehwa

2024-6-6 | by Chungmok
    from myalgorithm import algorithm를 try block 안으로 옮김(myalgorithm.py 가 없거나 algorithm() 함수가 없을 때 exception 발생)
'''

def run_algorithm(problem_file, timelimit, alg_func):

    with open(problem_file, 'r') as f:
        prob = json.load(f)

    K = prob['K']

    ALL_ORDERS = [Order(order_info) for order_info in prob['ORDERS']]
    ALL_RIDERS = [Rider(rider_info) for rider_info in prob['RIDERS']]

    DIST = np.array(prob['DIST'])
    for r in ALL_RIDERS:
        r.T = np.round(DIST/r.speed + r.service_time)

    alg_start_time = time.time()
    
    exception = None
    solution = None

    try:
        # Run algorithm!
        solution = alg_func(K, ALL_ORDERS, ALL_RIDERS, DIST, timelimit)
    except Exception as e:
        exception = f'{e}'
        print (exception)

    alg_end_time = time.time()

    # Re-read the problem due to possibility of any undisirable object modifications during the algorithm
    with open(problem_file, 'r') as f:
        prob = json.load(f)

    K = prob['K']

    ALL_ORDERS = [Order(order_info) for order_info in prob['ORDERS']]
    ALL_RIDERS = [Rider(rider_info) for rider_info in prob['RIDERS']]

    DIST = np.array(prob['DIST'])

    for r in ALL_RIDERS:
        r.T = np.round(DIST/r.speed + r.service_time)

    checked_solution = solution_check(K, ALL_ORDERS, ALL_RIDERS, DIST, solution)

    checked_solution['time'] = alg_end_time - alg_start_time
    checked_solution['timelimit_exception'] = (alg_end_time - alg_start_time) > timelimit + 1 # allowing additional 1 second!
    checked_solution['exception'] = exception

    checked_solution['prob_name'] = prob['name']
    checked_solution['prob_file'] = problem_file

    return checked_solution


def main(prob_file, timelimit):

    try:
        from myalgorithm import algorithm
        
        checked_solution = run_algorithm(problem_file=prob_file, timelimit=timelimit, alg_func=algorithm)
        # print(checked_solution)

        result_filename = checked_solution['prob_name'] 

        with open(result_filename + '__results__.json', 'w') as f:
            json.dump(checked_solution, f, indent=4)
            # pretty_json_str = pprint.pformat(checked_solution, compact=True, sort_dicts=False).replace("'",'"')
            # f.write(pretty_json_str)

        return 0

    except Exception as e:
        raise Exception('test_error465' + str(e)) # print(e) --> raise Exception('test_error465' + str(e)) by sehwa
        return 1

if __name__ == '__main__':

    main("D:\Project\Python\ogc_solutions\input\STAGE1_6.json", 120)



