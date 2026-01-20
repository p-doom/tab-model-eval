project_root="/home/hk-project-pai00039/tum_ind3695/projects/tab-model-eval/"
sandbox_dir_BAK="${project_root}/sandbox/eval_sandbox_BAK"
sandbox_dir="${project_root}/sandbox/eval_sandbox"
mkdir -p "$sandbox_dir"

target_file="${project_root}/data/eval/handcrafted/add_type_hints.md"


list_commands() {
  cat <<'EOF'
mypy --strict src/utils/math.py
cat -n src/utils/math.py
sed -i '7,7c\def calculate_area(radius: float) -> float:' src/utils/math.py && cat -n src/utils/math.py | sed -n '1,17p'
sed -i '12,12c\def calculate_circumference(radius: float) -> float:' src/utils/math.py && cat -n src/utils/math.py | sed -n '1,18p'
sed -i '17,17c\def calculate_volume(radius: float) -> float:' src/utils/math.py && cat -n src/utils/math.py | sed -n '3,18p'
mypy --strict src/utils/math.py
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
