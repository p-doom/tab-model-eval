``` bash
cat <<'EOF' > palindrome_checker.py
def is_palindrome(text):
    text = text.lower().replace(" ", "")
    return text == text[::-1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check if text is a palindrome")
    parser.add_argument("text", help="Text to check")
    args = parser.parse_args()
    if is_palindrome(args.text):
        print(f'"{args.text}" is a palindrome')
    else:
        print(f'"{args.text}" is not a palindrome')
EOF
```

``` bash
python palindrome_checker.py racecar
```

<stdout>
"racecar" is a palindrome
</stdout>

``` bash
python palindrome_checker.py hello
```

<stdout>
"hello" is not a palindrome
</stdout>