project_root="/home/hk-project-pai00039/tum_ind3695/projects/tab-model-eval/"
sandbox_dir_BAK="${project_root}/sandbox/eval_sandbox_BAK"
sandbox_dir="${project_root}/sandbox/eval_sandbox"
mkdir -p "$sandbox_dir"

target_file="${project_root}/data/eval/handcrafted/add_import_easy.md"

list_commands() {
  cat <<'EOF'
# python src/input_pipeline/validation_to_testcases.py --input_file data/temp/validation.jsonl --output_file data/temp/validation_testcases.jsonl
cat -n src/input_pipeline/validation_to_testcases.py
sed -i '1i\import json' src/input_pipeline/validation_to_testcases.py && cat -n src/input_pipeline/validation_to_testcases.py | sed -n '1,11p'
python src/input_pipeline/validation_to_testcases.py --input_file data/temp/validation.jsonl --output_file data/temp/validation_testcases.jsonl
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

