# Semantic Equivalence Evaluation

Determine if the **Generated Command** achieves the same outcome as the **Expected Command**.

## Context

You are evaluating a tab-completion model that predicts the next bash command a developer would execute. The model is trained to follow specific command formats (particularly line-based sed edits), but equivalent approaches that achieve the same outcome should be considered correct.

## Evaluation Reasoning Framework

You are given:
1. **Conversation History**: Shows what the user was doing and what problems arose
2. **Expected Command**: The reference solution
3. **Generated Command**: The model's prediction to evaluate

### Step-by-Step Reasoning

**Step 1 - Understand Context**: 
Read the conversation history. What task is the user performing? What state is the system in? What problem needs to be solved?

**Step 2 - Identify Intent**: 
What is the PURPOSE of the Expected Command? Why would a user run this command given the history?

**Step 3 - Separate Critical from Flexible**:
- **Critical**: Elements that MUST match (target files, core operation, essential flags)
- **Flexible**: Elements that CAN vary (messages, comments, display commands, syntax style)

**Step 4 - Evaluate Generated Command**:
Does it achieve the same PURPOSE? Would the system be in a functionally equivalent state after execution?

**Step 5 - Apply Leniency**:
- If differences are only in flexible elements → Equivalent
- If an extra helpful element is added (comment, better message) → Equivalent
- If the core operation or target differs → Not Equivalent

## Definitions

**Semantically Equivalent** — Commands produce the same functional result when executed.

### Differences that DO NOT affect equivalence:

**Command Syntax**
- Flag order (`ls -la` ≡ `ls -al`)
- Short vs long flags (`-h` ≡ `--help`)
- Equivalent syntax (`cat file | grep x` ≡ `grep x file`)
- Whitespace or quoting style
- Path style: relative vs absolute paths resolving to the same location

**File Editing**
- Different sed/awk/perl approaches achieving the same file state
- Line-based replacement vs substitution producing same result:
  - `sed -i '6c\new content' file` ≡ `sed -i 's/old content/new content/' file` (if same outcome)
- Viewport/display commands appended after the main operation (`&& cat -n file | sed -n '1,10p'`)
- Line insertion position when order doesn't matter (e.g., import statements in most languages)
- Minor formatting differences in generated files that don't affect program behavior

**Git Commands**
- Commit message text (any reasonable, meaningful message is valid)
- Branch name variations (when creating new branches, if reasonable)
- Tag name variations (if reasonable)

**General**
- Echo/print statement text variations (unless exact output is the task goal)
- Additional helpful comments in generated code
- Tool choice for same operation (`cat` vs `less` vs `head` for viewing files)

### NOT Equivalent — Commands produce different results:

- Different target files, directories, or resources
- Missing or extra flags that change behavior significantly
- Different operations entirely
- Malformed or incomplete commands
- One command would fail while the other succeeds
- Order differences when order semantically matters (e.g., command pipelines, ordered data insertions)
- Missing critical fixes that the history indicates are needed

## Intent-Based Flexibility by Task Category

| Intent Category | Core Requirement | Flexible Elements |
|-----------------|------------------|-------------------|
| **Fix an error** | Resolves the error shown in history | Exact fix approach, comments, formatting, sed syntax |
| **Commit changes** | Commits the correct staged files | Commit message text |
| **Edit file content** | Final file state achieves intended change | Line numbers (if content correct), viewport display, sed style |
| **Navigate/explore** | Views the correct file/location | Tool choice (cat/less/head/bat), exact line ranges |
| **Run/test code** | Executes the intended script/test | Interpreter variations (python/python3) |
| **Git workflow** | Performs correct git action | Branch names, message text, flag order |
| **Install/setup** | Installs correct packages/dependencies | Package manager flags, version specifiers |

## User Acceptance Test

When uncertain, ask yourself:

1. **Task Completion**: Would the user consider their task complete after running the Generated Command?
2. **Same System State**: Would the repository/file system be in a functionally equivalent state?
3. **No Harmful Differences**: Does the Generated Command avoid breaking functionality, losing data, or missing the actual goal?
4. **Reasonable Variation**: Is this a difference a developer might naturally make?

**If all answers are YES → Equivalent (1)**

## Examples

### Example 1: Git Commit (EQUIVALENT)
**History**: User created files, ran `git add data/eval/handcrafted/`
**Expected**: `git commit -m "created more examples"`
**Generated**: `git commit -m "Add handcrafted tasks for git commit, pull, and push"`

**Reasoning**: Intent is to commit the staged files. Both commands perform `git commit -m` with a valid, meaningful message. The message text is a flexible element. → **Equivalent (1)**

### Example 2: Fix Import Error (EQUIVALENT)
**History**: ImportError for `load_config` from `utils`, found function in `src/common/config_utils.py`
**Expected**: `sed -i '6,6c\from common.config_utils import load_config' src/main.py && cat -n src/main.py | sed -n '1,10p'`
**Generated**: `sed -i 's/from utils import load_config/from common.config_utils import load_config/' src/main.py`

**Reasoning**: Intent is to fix the import on line 6. Both commands change the import to `common.config_utils`. The viewport display (`&& cat -n ...`) is optional. Same import path, same target file. → **Equivalent (1)**

### Example 3: Different Import Path (NOT EQUIVALENT)
**Expected**: `sed -i '6c\from common.config_utils import load_config' src/main.py`
**Generated**: `sed -i 's/from utils import load_config/from src.common.config_utils import load_config/' src/main.py`

**Reasoning**: The import paths differ: `common.config_utils` vs `src.common.config_utils`. These may resolve to different modules depending on Python path configuration. This is a critical element that affects whether the fix works. → **Not Equivalent (0)**

### Example 4: Same Edit, Different Syntax (EQUIVALENT)
**Expected**: `sed -i '15,17c\    return x + y' src/math.py`
**Generated**: `sed -i '15,17d; 15i\    return x + y' src/math.py`

**Reasoning**: Both commands result in lines 15-17 being replaced with `return x + y`. Different sed syntax, same file outcome. → **Equivalent (1)**

### Example 5: Wrong Target File (NOT EQUIVALENT)
**History**: Error in `src/main.py`
**Expected**: `sed -i '10c\import logging' src/main.py`
**Generated**: `sed -i '10c\import logging' src/utils.py`

**Reasoning**: Different target files. The Generated Command edits `utils.py` instead of `main.py`. → **Not Equivalent (0)**

## Conversation History

{context}

## Commands

**Expected:**
{expected}

**Generated:**
{generated}

## Task

Analyze the conversation history to understand the task intent, then compare the commands.

Respond with this exact JSON:

{{
  "reason": "<Step 1: Intent from history. Step 2: Critical elements. Step 3: Whether generated achieves same outcome>",
  "equivalent": <1 or 0>
}}

Where:
- `1` = Same outcome (semantically equivalent)
- `0` = Different outcome (not equivalent)

**Remember**: Focus on FUNCTIONAL equivalence. If both commands achieve the user's goal from the conversation history, they are equivalent—even with different syntax, messages, or approaches.