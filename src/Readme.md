*This project has been created as part of the 42 curriculum by [rydelepi].*

---

# LLM Function Calling Engine

## Description

This project implements a **constrained decoding engine** that forces a small language model (LLM) to generate syntactically valid JSON function calls — every time, without post-processing hacks or retries.

The core idea is to intercept the model's token-by-token generation process and mask out any token that would lead to an invalid output. The result is a system that takes a natural language prompt and reliably produces a structured JSON object matching one of the pre-defined function schemas.

**Goal:** Given a set of function definitions and a user prompt, generate a valid JSON function call of the form:
```json
{"name": "function_name", "parameters": {"param1": value1, ...}}
```

The three main components are:
- `models.py` — Pydantic schemas for functions, prompts, and results
- `decoder.py` — The constrained decoding logic
- `__main__.py` — CLI entry point orchestrating the full pipeline

---

## Instructions

### Requirements

- Python 3.10+
- `pydantic`
- `numpy`
- `llm_sdk` (provides `Small_LLM_Model`)

Install dependencies:
```bash
pip install pydantic numpy
# Install llm_sdk according to your project environment
```

### Project Structure

```
.
├── __main__.py
├── src/
│   ├── models.py
│   └── decoder.py
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/
│       └── function_calling_results.json
```

### Running the Program

```bash
python -m src [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--functions_definition` | `data/input/functions_definition.json` | Path to the JSON file defining available functions |
| `--input` | `data/input/function_calling_tests.json` | Path to the JSON file containing test prompts |
| `--output` | `data/output/function_calling_results.json` | Path where results will be saved |

**Example:**
```bash
python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

---

## Example Usage

### Input: `functions_definition.json`

```json
[
  {
    "name": "get_weather",
    "description": "Returns the current weather for a given city.",
    "parameters": {
      "city": { "type": "string" },
      "unit": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

### Input: `function_calling_tests.json`

```json
[
  { "prompt": "What is the weather in Paris in Celsius?" },
  { "prompt": "Give me the temperature in Tokyo in Fahrenheit." }
]
```

### Output: `function_calling_results.json`

```json
[
  {
    "prompt": "What is the weather in Paris in Celsius?",
    "name": "get_weather",
    "parameters": { "city": "Paris", "unit": "Celsius" }
  },
  {
    "prompt": "Give me the temperature in Tokyo in Fahrenheit.",
    "name": "get_weather",
    "parameters": { "city": "Tokyo", "unit": "Fahrenheit" }
  }
]
```

---

## Algorithm Explanation

### Constrained Decoding

Standard LLM generation samples or greedily picks the most likely next token from the full vocabulary at each step. This works well for free-form text but is unreliable for structured formats — the model may generate invalid JSON, hallucinate function names, or produce malformed outputs.

**Constrained decoding** solves this by modifying the logits (raw scores) before each token is chosen:

1. **Vocabulary filtering** — At startup, the full model vocabulary is loaded and filtered to only ASCII-printable tokens. This reduces the search space and avoids encoding issues.

2. **Prefix validation** — At each generation step, for every candidate token, the engine checks whether appending that token to the already-generated text produces a **valid prefix** of the expected output format. Tokens that would make the prefix invalid are masked to `-inf` (impossible to select).

3. **Greedy selection** — Among all still-valid tokens, the one with the highest logit score is selected. This is greedy decoding (`argmax`), not sampling.

4. **Termination** — Generation stops when the number of open braces `{` equals the number of closing braces `}`, meaning the JSON object is complete.

### Prefix Grammar

The `is_valid_prefix` method enforces this grammar step by step:

```
{"name":"<FUNCTION_NAME>","parameters":{<...>}
```

- The prefix must start with `{"name":"`
- The function name must be a prefix of (or exactly) one of the allowed function names
- After the function name, the next expected segment is `,"parameters":{`
- After that, any content is currently allowed (parameters are unconstrained)

---

## Design Decisions

**Greedy decoding over sampling:** Using `argmax` instead of temperature sampling guarantees determinism and avoids the risk of a low-probability token breaking the grammar constraint. For function calling, correctness matters more than diversity.

**ASCII-only vocabulary filter:** Tokens containing non-ASCII characters are excluded early. This avoids decoding errors and keeps the constraint-checking loop fast, since the valid vocabulary is significantly smaller than the full one.

**Prefix-based validation instead of full grammar parsing:** A full context-free grammar parser would be more expressive but much slower. Checking prefixes with string operations is O(n) per token and sufficient for the fixed output schema used here.

**Pydantic for schema validation:** After generation, the output is parsed and validated with Pydantic (`FunctionCallResult`). This provides a clean second layer of validation beyond the syntactic constraint, catching semantic issues (e.g. wrong field types).

**Partial result saving on interruption:** If the user interrupts execution (`Ctrl+C`), results generated so far are saved to the output file rather than lost. This is important for long test batches.

---

## Performance Analysis

**Accuracy:** The constrained decoder guarantees that every generated output is at minimum a syntactically valid prefix of the expected format. Whether the *function name* and *parameters* are semantically correct depends on the underlying LLM's understanding of the prompt. For well-prompted small models, function name selection is generally reliable; parameter value quality depends on model capability.

**Speed:** The main bottleneck is the per-step vocabulary scan: for each of the `max_tokens` steps, every token in the filtered vocabulary is tested with `is_valid_prefix`. With a filtered vocabulary of ~10,000–30,000 tokens and `max_tokens=150`, this is roughly 1.5–4.5 million string operations per prompt. This is fast in practice (milliseconds per step on CPU) but does not scale to very large vocabularies without further optimization (e.g. a trie).

**Reliability:** The brace-counting termination condition is simple but can fail in edge cases where the model generates strings containing `{` or `}` as content within parameter values. A more robust approach would use a JSON parser's streaming state.

---

## Challenges Faced

**Vocabulary decoding inconsistencies:** Some tokenizers produce tokens that decode to multi-byte sequences, empty strings, or raise exceptions. The `_build_and_filter_vocab` method handles this with a `try/except` and a non-empty check to silently skip problematic tokens.

**Input ID format variability:** The `model.encode()` method may return either a flat list or a batch (list of lists). The decoder handles both cases with a check `if raw_ids and isinstance(raw_ids[0], list)`.

**Dead-end states (impasse):** It is possible for the constraint to eliminate all tokens (e.g. if the model's top tokens are all illegal at a given point). The decoder detects this when `masked_logits[argmax] == -inf` and exits with an error message rather than hanging or crashing.

**Balancing strictness and completeness:** The prefix grammar is strict for the function name (must match an allowed name exactly) but lenient for parameters (any content after `"parameters":{` is accepted). This was a pragmatic tradeoff: enforcing parameter schemas would require knowing their types and allowed values at decode time, which significantly complicates the grammar.

---

## Testing Strategy

Testing was performed at two levels:

**Unit-level (prefix validation):** The `is_valid_prefix` method was tested with a range of hand-crafted strings:
- Valid prefixes at each stage of the grammar (`{`, `{"name":"`, `{"name":"get_weather`, etc.)
- Invalid prefixes (wrong start, spaces in function name, unknown function name)
- Edge cases (empty string, prefix exactly at a boundary)

**Integration-level (full pipeline):** The full pipeline was run against a set of test prompts in `function_calling_tests.json`. Results were checked against expected outputs:
- JSON validity (`json.loads` must succeed)
- Pydantic schema compliance (`FunctionCallResult` must validate)
- Correct function name selection for unambiguous prompts
- Graceful handling of ambiguous or out-of-scope prompts (the model may still pick the closest function)

**Regression testing:** After any change to `is_valid_prefix` or the prompt template, the full test set was re-run to check for regressions in valid-output rate.

---

## Resources

### Documentation & References

- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/) — data validation and schema enforcement
- [NumPy Documentation](https://numpy.org/doc/) — logit manipulation and `argmax`

### AI Usage

AI assistance (Claude, Anthropic) was used for the following tasks during this project:

- **Debugging** the vocabulary filtering loop, specifically handling tokenizer edge cases (empty decodes, encoding exceptions)
- **Reviewing** the `is_valid_prefix` logic for correctness at each grammar stage
- **Writing and reviewing** Pydantic model definitions in `models.py`
- **Generating this README** from the project's source code and architecture