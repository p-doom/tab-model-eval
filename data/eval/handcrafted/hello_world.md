# Assistant
``` bash
ls -la
```

# User
<stdout>
[alfred.nguyen@hai-login1.haicore.berlin:~] $ ls -la
total 336
drwxrwx--- 3 alfred.nguyen alfred.nguyen   318 Nov 13 18:05 .
drwxr-xr-x 9 alfred.nguyen alfred.nguyen   171 Nov 12 15:29 ..
[alfred.nguyen@hai-login1.haicore.berlin:~] $ 
</stdout>

# Assistant
``` bash
cat <<'EOF' > hello_world.py
print("Hello, World!")
EOF && cat -n hello_world.py
```

# User
<stdout>
     1  print("Hello, World!")
</stdout>

# Assistant
```bash
python hello_world.py
```

# User
<stdout>
Hello, World!
</stdout>

# Assistant
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```