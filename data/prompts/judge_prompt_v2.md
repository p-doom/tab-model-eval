# Semantic Equivalence Evaluation

Determine if the **Generated Command** achieves the same outcome as the **Expected Command**.

## Definitions

**Semantically Equivalent** — Commands produce the same result when executed. Differences that DO NOT affect equivalence:
- Flag order (`ls -la` ≡ `ls -al`)
- Short vs long flags (`-h` ≡ `--help`)
- Equivalent syntax (`cat file | grep x` ≡ `grep x file`)
- Whitespace or quoting style

**NOT Equivalent** — Commands produce different results:
- Different files, directories, or arguments
- Missing or extra flags that change behavior
- Different operations entirely
- Malformed or incomplete commands

## Commands

**Expected:**
{expected}

**Generated:**
{generated}

## Task

Compare the commands and respond with this exact JSON:

{{
  "equivalent": <1 or 0>,
  "reason": "<brief explanation>"
}}

Where:
- `1` = Same outcome (semantically equivalent)
- `0` = Different outcome (not equivalent)