# Semantic Equivalence Evaluation

Determine if the **Generated Command** achieves the same outcome as the **Expected Command**. Consider the **General Guidelines** and **Assertions** when evaluating.

## Definitions

**Semantically Equivalent** — Commands produce the same result when executed. Differences that DO NOT affect equivalence:
- Flag order (`ls -la` ≡ `ls -al`)
- Short vs long flags (`-h` ≡ `--help`)
- Equivalent syntax (`cat file | grep x` ≡ `grep x file`)
- Whitespace or quoting style
- Line insertion position when order doesn't matter (e.g., import statements in most languages)
- Minor formatting differences in generated files that don't affect program behavior
- Different commit messages as long as they make sense
- Different file/directory/variable naming as long as it doesn't affect the program/workflow
- Printing a few extra lines

**NOT Equivalent** — Commands produce different results:
- Violating the Assertions
- Different files, directories, or arguments
- Missing or extra flags that change behavior
- Different operations entirely
- Malformed or incomplete commands
- Order differences only when order semantically matters (e.g., command pipelines, ordered data insertions)
- Deleting or modifying different lines

**General Guidelines:**
We only accept three types of commands:

1) File viewing
We specfically only accept the `cat -n FILE` command to view the contents of a file. No other commands are allowed.

2) File edits
We specfically only accept sed commands to edit files. No other commands are allowed.
After editing a file, the file content around the edited region should be displayed using `cat -n FILE | sed -n 'START,ENDp'`.
The region is defined by a viewport which is roughly 10 lines before and after the edited region.
The viewport might be smaller when the edited region is near the edges of the file (beginning or end of the file).
Make sure edits are done in one sed command and not multiple sed commands.

3) Running scripts
We accept arbitrary bash commands to run scripts, tests, git commands, tools, debugging commands, etc. 


## Conversation History

{context}

## Commands

**Expected:**
{expected}

**Generated:**
{generated}

**Assertions:**
{assertions}

## Task

Compare the commands and respond with this exact JSON:

{{
  "reason": "<brief reasoning>",
  "equivalent": <1 or 0>
}}

Where:
- `1` = Same outcome (semantically equivalent)
- `0` = Different outcome (not equivalent)
# Semantic Equivalence Evaluation

Determine if the **Generated Command** achieves the same outcome as the **Expected Command**. Consider the **General Guidelines** and **Assertions** when evaluating.

## Definitions

**Semantically Equivalent** — Commands produce the same result when executed. Differences that DO NOT affect equivalence:
- Flag order (`ls -la` ≡ `ls -al`)
- Short vs long flags (`-h` ≡ `--help`)
- Equivalent syntax (`cat file | grep x` ≡ `grep x file`)
- Whitespace or quoting style
- Line insertion position when order doesn't matter (e.g., import statements in most languages)
- Minor formatting differences in generated files that don't affect program behavior
- Different commit messages as long as they make sense
- Different file/directory/variable naming as long as it doesn't affect the program/workflow
- Printing a few extra lines

**NOT Equivalent** — Commands produce different results:
- Violating the Assertions
- Different files, directories, or arguments
- Missing or extra flags that change behavior
- Different operations entirely
- Malformed or incomplete commands
- Order differences only when order semantically matters (e.g., command pipelines, ordered data insertions)
- Deleting or modifying different lines

**General Guidelines:**
- When editing files, only the sed command is to be used. No other commands are allowed.
- After editing a file, the file contents should be catted with a viewport using `cat -n FILE | sed -n 'START,ENDp'`.
- The viewport should be roughly 10 lines before and after the edited region (they dont have to be exactly 10 lines)
- Only use a single sed command to edit the file. Do not use multiple sed commands.
- Generally, only one single command should be used
  - e.g. when running scripts, tests, git commands, tools, etc.
  - except for sed command where the viewport is required

## Conversation History

{context}

## Commands

**Expected:**
{expected}

**Generated:**
{generated}

**Assertions:**
{assertions}

## Task

Compare the commands and respond with this exact JSON:

{{
  "reason": "<brief reasoning>",
  "equivalent": <1 or 0>
}}

Where:
- `1` = Same outcome (semantically equivalent)
- `0` = Different outcome (not equivalent)