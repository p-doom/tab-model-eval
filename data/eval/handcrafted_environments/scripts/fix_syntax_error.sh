project_root="/home/hk-project-pai00039/tum_ind3695/projects/tab-model-eval/"
sandbox_dir_BAK="${project_root}/sandbox/eval_sandbox_BAK"
sandbox_dir="${project_root}/sandbox/eval_sandbox"
rm -rf "$sandbox_dir"
mkdir -p "$sandbox_dir"

target_file="${project_root}/data/eval/handcrafted/fix_syntax_error.md"


list_commands() {
  cat <<'EOF'
python src/train.py
cat -n src/train.py
sed -i '17,17c\    print(f"Training started at {time.time()}")' src/train.py && cat -n src/train.py | sed -n '6,27p'
# python src/train.py
EOF
}


echo "Copying $sandbox_dir_BAK to $sandbox_dir"
cp -r "$sandbox_dir_BAK"/* "$sandbox_dir"
echo "Copied $sandbox_dir_BAK to $sandbox_dir"
echo "================================================"
echo "================================================"


echo "Changing to $sandbox_dir"
cd "$sandbox_dir"

echo "Writing to $target_file"
(list_commands | while IFS= read -r command; do
    echo "# Assistant <NO_EVAL>"
    echo '```bash'
    echo "$command"
    echo '```'
    echo
    echo "# User"
    echo "<stdout>"
    eval "$command"
    echo "</stdout>"
    echo
done ) > "${target_file}"
