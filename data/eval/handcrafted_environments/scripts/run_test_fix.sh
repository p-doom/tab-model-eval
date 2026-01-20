project_root="/home/hk-project-pai00039/tum_ind3695/projects/tab-model-eval/"
sandbox_dir_BAK="${project_root}/sandbox/eval_sandbox_BAK"
sandbox_dir="${project_root}/sandbox/eval_sandbox"
mkdir -p "$sandbox_dir"

target_file="${project_root}/data/eval/handcrafted/run_test_fix.md"

list_commands() {
  cat <<'EOF'
pytest tests/test_math.py
cat -n tests/test_math.py
sed -i '15,15c\    assert add(2, 2) == 4' tests/test_math.py && cat -n tests/test_math.py | sed -n '1,21p'
pytest tests/test_math.py
EOF
}



echo "Copying $sandbox_dir_BAK to $sandbox_dir"
cp -r "$sandbox_dir_BAK"/* "$sandbox_dir"
echo "Copied $sandbox_dir_BAK to $sandbox_dir"
echo "================================================"
echo "================================================"


echo "Changing to $sandbox_dir"
cd "$sandbox_dir"
export PYTHONPATH=$PYTHONPATH:$sandbox_dir

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

