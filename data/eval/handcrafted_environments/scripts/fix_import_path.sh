project_root="/home/hk-project-pai00039/tum_ind3695/projects/tab-model-eval/"
sandbox_dir_BAK="${project_root}/sandbox/eval_sandbox_BAK"
sandbox_dir="${project_root}/sandbox/eval_sandbox"
mkdir -p "$sandbox_dir"

target_file="${project_root}/data/eval/handcrafted/fix_import_path.md"


list_commands() {
  cat <<'EOF'
python src/main.py
cat -n src/common/config_utils.py
cat -n src/main.py
sed -i '6,6c\from common.config_utils import load_config' src/main.py && cat -n src/main.py | sed -n '1,16p'
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
done ) | tee "${target_file}"
