``` bash
ls -la
```

<stdout>
[alfred.nguyen@hai-login1.haicore.berlin:~] $ ls -la
total 336
drwxrwx--- 3 alfred.nguyen alfred.nguyen   318 Nov 13 18:05 .
drwxr-xr-x 9 alfred.nguyen alfred.nguyen   171 Nov 12 15:29 ..
[alfred.nguyen@hai-login1.haicore.berlin:~] $ 
</stdout>

``` bash
cat <<'EOF' > hello_world.py
print("Hello, World!")
EOF

```
``` bash
cat -n hello_world.py
```

<stdout>
     1  print("Hello, World!")
</stdout>

```bash
python hello_world.py
```

<stdout>
Hello, World!
</stdout>
