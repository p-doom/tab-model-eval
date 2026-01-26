# Assistant <NO_EVAL>
```bash
gi push
```

# User
<stdout>
bash: gi: command not found
</stdout>

# Assistant <EVAL>
```bash
git push
```

<assertions>
Make sure that:
- we push the changes to the remote repository
- similar push commands that specify the branch (dev/ali) are also valid
- nothing else is modified at all
</assertions>

# User
<stdout>
Everything up-to-date
</stdout>
