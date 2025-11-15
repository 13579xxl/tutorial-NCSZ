#斐波那契数列第n项的实现
#可在学完函数后一同学习
def Fibonacci_sequence(n):
    if n <= 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    a, b = 1, 1
    #使用了range函数，表示从3开始到n+1次结束，因为1和2我们已经考虑了
    for i in range(3, n + 1):
        a, b = b, a + b
# 等价于：
# a = b   
# b = a + b
# 因为该等式的执行顺序是先计算右边的量再赋值给左边对应量，属于比较简洁的写法 
    return b
print(Fibonacci_sequence(10))
    