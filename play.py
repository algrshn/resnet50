from multiprocessing import Pool

class Engine(object):
    def __init__(self, num_of_threads, arg1):
        self.num_of_threads = num_of_threads
        self.arg1 = arg1
    def __call__(self, thread_num):
        
        for i in range(self.arg1):
            
            if(i % self.num_of_threads == thread_num):
                print("thread_num = {} | i = {}".format(thread_num,i))
        
        

try:
    pool = Pool(3) 
    engine = Engine(3, 1000)
    data_outputs = pool.map(engine, [0,1,2])
finally: # To make sure processes are closed in the end, even if errors happen
    pool.close()
    pool.join()
