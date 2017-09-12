# 程序参考：http://www.cnblogs.com/284628487a/p/5590857.html
from multiprocessing import Process, Pool
import os
import time

# 编写多进程的第一步就是先将子进程要执行的代码编成函数。不同方式接收参数的方式也不一样。
### 方式1：apply_async
# def run_proc(name, sleep_time=1):

### 方式2：map方式 ,缺点貌似是子进程函数的输入参数没法少，默认的也要输入。
def run_proc(process_input):
    name = process_input[0]
    sleep_time = int(process_input[1])

    # 剩下的都一样。
    print('Child %s process (%s) is running.' % (name, os.getpid()))
    # 干点事情吧
    time.sleep(sleep_time)
    print('Child %s process (%s) is End.' % (name, os.getpid()))
    return name


# 注意多进程一定要写判断，否则会无限循环。
if __name__ == '__main__':
    # 父进程的id
    print('Parent process %s.' % os.getpid())

    finished_child = []

    # 创建资源池，如果子进程比较多的话，这样比较方便。
    p = Pool(10)

    ####创建方式1：apply_async
    # for i in range(10):
    #     ## 若是非进程池的执行方式：
    #     p = Process(target=run_proc, args=('test'+str(i),i))
    #     p.start()     # 非进程池，需要手工start。
    #     ## 若是进程池的执行方式：
    #     finished_child.append(p.apply_async(func=run_proc, args=('test' + str(i), i)))  # 进程池不需要start，自动start。


    #### 创建方式2：map方式： 如果只是要接收返回值，又不想用队列共享变量，那么简单的用：
    process_result = p.map(func=run_proc,
                           iterable=(['test' + str(i),i ] for i in range(10)))  # 注意输入参数必须在iterable中。map的缺点貌似是子进程函数的输入参数没法少，默认的也要输入。
    # @注意了，返回值的所有进程的返回值都写到一个列表中！不注意的话就坑爹啊！
    # finished_child.append(process_result)
    # 本来是计划append()每个结果，但实际上就是所有结果啊。
    finished_child = process_result

    # 剩下的都一样
    print('All Child process will start.')

    p.close()  # 进程池的方式下，如果要join()，必须先close()，防止添加新的子进程。
    p.join()  # 有了join()，父进程会等待所有子进程都结束后才继续往下执行。 所有进程资源都是在join()之后才一起释放，所以windows下比较占用资源。

    print('All Child process should be ended.')
    print('结束的进程：')
    print(finished_child)
