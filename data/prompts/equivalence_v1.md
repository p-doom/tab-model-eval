You are evaluating bash commands in a conversational assistant context.

Context (conversation history):
{json.dumps(context, indent=2)}

Expected Command:
{expected}

Generated Command:
{generated}

Task: Determine if the generated command is semantically equivalent to the expected command.
Two commands are semantically equivalent if they achieve the same goal, even if the exact syntax differs.

Respond in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<brief explanation>"
}}

Score Guidelines:
- 1.0: Commands are functionally identical or achieve the exact same result
- 0.7-0.9: Commands achieve the same goal with minor differences (e.g., different flags, equivalent syntax)
- 0.4-0.6: Commands are related but have meaningful differences
- 0.0-0.3: Commands are significantly different or incorrect