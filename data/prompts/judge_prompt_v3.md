# Semantic Equivalence Evaluation

Determine if the **Generated Command** achieves the same outcome as the **Expected Command**.

## Definitions

**Semantically Equivalent** — Commands produce the same result when executed. Differences that DO NOT affect equivalence:
- Flag order (`ls -la` ≡ `ls -al`)
- Short vs long flags (`-h` ≡ `--help`)
- Equivalent syntax (`cat file | grep x` ≡ `grep x file`)
- Whitespace or quoting style
- Line insertion position when order doesn't matter (e.g., import statements in most languages)
- Minor formatting differences in generated files that don't affect program behavior

**NOT Equivalent** — Commands produce different results:
- Different files, directories, or arguments
- Missing or extra flags that change behavior
- Different operations entirely
- Malformed or incomplete commands
- Empty commands
- Order differences only when order semantically matters (e.g., command pipelines, ordered data insertions)

## Commands

**Expected:**
{expected}

**Generated:**
{generated}

## Task

Compare the commands and respond with this exact JSON:

{{
  "reason": "<brief reasoning>",
  "equivalent": <1 or 0>
}}

Where:
- `1` = Same outcome (semantically equivalent)
- `0` = Different outcome (not equivalent)