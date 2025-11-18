``` bash
cat <<'EOF' > calculator.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple calculator")
    parser.add_argument("operation", choices=["add", "subtract", "multiply", "divide"])
    parser.add_argument("a", type=float)
    parser.add_argument("b", type=float)
    args = parser.parse_args()

    if args.operation == "add":
        result = add(args.a, args.b)
    elif args.operation == "subtract":
        result = subtract(args.a, args.b)
    elif args.operation == "multiply":
        result = multiply(args.a, args.b)
    else:
        result = divide(args.a, args.b)

    print(result)
EOF
```

``` bash
python calculator.py add 5 3
```

<stdout>
8.0
</stdout>

``` bash
python calculator.py multiply 4 7
```

<stdout>
28.0
</stdout>
