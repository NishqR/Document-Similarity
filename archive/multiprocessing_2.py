import multiprocessing
import time

def square(out_list):
    for i in range(1000000000*100000000000):
        out_list.append(random.random())

def par():
    for num_cores in range(1,multiprocessing.cpu_count()+1):

        start = time.time()
        pool = multiprocessing.Pool(processes=num_cores)
        #x = range(0,1000000)

        out_list = []
        results = pool.map(square,out_list)
        pool.close()
        pool.join()
        stop = time.time()
        print(num_cores,stop-start)

if __name__=='__main__':
    par()