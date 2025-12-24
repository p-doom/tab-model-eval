# Assistant
```bash
ls -la
```

# User
<stdout>
total 336
drwxrwx--- 3 alfred.nguyen alfred.nguyen   318 Nov 13 18:05 .
drwxr-xr-x 9 alfred.nguyen alfred.nguyen   171 Nov 12 15:29 ..
</stdout>

# Assistant
```bash
cat <<'EOF' > fibonnacci.py
import argparse

def fibonnacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonnacci(n-1) + fibonnacci(n-2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    print(fibonnacci(args.n))
EOF && cat -n fibonnacci.py
```

# User
<stdout>
     1  import argparse
     2  def fibonnacci(n):
     3      if n <= 0:
     4          return 0
     5      elif n == 1:
     5          return 1
     7      else:
     8          return fibonnacci(n-1) + fibonnacci(n-2)
     9 
    10 if __name__ == "__main__":
    11     parser = argparse.ArgumentParser()
    12     parser.add_argument("--n", type=int, default=10)
    13     args = parser.parse_args()
    14     print(fibonnacci(args.n))
</stdout>

# Assistant
```bash
python fibonnacci.py --n 15
```

# User
<stdout>
55
</stdout>
