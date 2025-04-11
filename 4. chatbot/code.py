def generate_fibonacci(n):
    fib_sequence = []

    if n < 0:
        return []

    a, b = 0, 1

    for _ in range(2, n + 1):  # Start from 2 to reach up to the nth number (inclusive)
        fib_sequence.append(a)
        a, b = a + b, a

    if len(fib_sequence) > 0 and fib_sequence[-1] < n:
        fib_sequence.pop()

    return fib_sequence

# Generate Fibonacci series up to 5
print(generate_fibonacci(100))