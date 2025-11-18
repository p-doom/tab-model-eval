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
    return a // b

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

``` bash
python calculator.py divide 5 2
```

<stdout>
2
</stdout>

``` bash
cat -n calculator.py
```


<stdout>
1 def add(a, b):
2 return a + b
3
4 def subtract(a, b):
5 return a - b
6
7 def multiply(a, b):
8 return a * b
9
11 if b == 0:
12 return "Error: Division by zero"
13 return a // b
14
15 if name == "main":
16 import argparse
17 parser = argparse.ArgumentParser(description="Simple calculator")
18 parser.add_argument("operation", choices=["add", "subtract", "multiply", "divide"])
19 parser.add_argument("a", type=float)
20 parser.add_argument("b", type=float)
21 args = parser.parse_args()
22
23 if args.operation == "add":
24 result = add(args.a, args.b)
25 elif args.operation == "subtract":
26 result = subtract(args.a, args.b)
27 elif args.operation == "multiply":
28 result = multiply(args.a, args.b)
29 else:
30 result = divide(args.a, args.b)
31
32 print(result)
</stdout>

``` bash
sed -i 's/^    return a \/\/ b/    return a \/ b/' calculator.py && cat -n calculator.py
```

<stdout>
     1  def add(a, b):
     2      return a + b
     3
     4  def subtract(a, b):
     5      return a - b
     6
     7  def multiply(a, b):
     8      return a * b
     9
    10  def divide(a, b):
    11      if b == 0:
    12          return "Error: Division by zero"
    13      return a / b
    14
    15  if __name__ == "__main__":
    16      import argparse
    17      parser = argparse.ArgumentParser(description="Simple calculator")
    18      parser.add_argument("operation", choices=["add", "subtract", "multiply", "divide"])
    19      parser.add_argument("a", type=float)
    20      parser.add_argument("b", type=float)
    21      args = parser.parse_args()
    22
    23      if args.operation == "add":
    24          result = add(args.a, args.b)
    25      elif args.operation == "subtract":
    26          result = subtract(args.a, args.b)
    27      elif args.operation == "multiply":
    28          result = multiply(args.a, args.b)
    29      else:
    30          result = divide(args.a, args.b)
    31
    32      print(result)
</stdout>

``` bash
python calculator.py divide 5 2
```
<stdout>
2.5
</stdout>
