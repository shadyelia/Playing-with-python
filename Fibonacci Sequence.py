from functools import lru_cache


def fibonacciRecursion(n):
    if n == 1:
        return 1
    elif n == 2:
        return 1
    elif n > 2:
        return fibonacciRecursion(n-1) + fibonacciRecursion(n-2)


fibonacci_cache = {}


def fibonacciImplementExplicity(n):
    if n in fibonacci_cache:
        return fibonacci_cache[n]

    if n == 1:
        value = 1
    elif n == 2:
        value = 1
    elif n > 2:
        value = fibonacciImplementExplicity(
            n-1) + fibonacciImplementExplicity(n-2)

    fibonacci_cache[n] = value

    return value


@lru_cache(maxsize=1000)
def fibonacciBuildPythonTool(n):
    if n == 1:
        return 1
    elif n == 2:
        return 1
    elif n > 2:
        return fibonacciRecursion(n-1) + fibonacciRecursion(n-2)


for n in range(1, 501):
    print(n, ":", fibonacciImplementExplicity(n))
