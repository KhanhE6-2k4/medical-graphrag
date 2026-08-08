# Manual

## Retrieval Evaluation

```bash
python -m Evaluation.retrieval_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --data_file ./evaluate_data/retrieval_eval_data.json \
  --output_file ./evaluate_result/retrieval_eval_result.json
```

## Answer Generation

```bash
python3 -m Generation.answer_generation_base \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --data_file ./Generation/input/generation_input_data.json \
  --output_file ./Generation/output/generation_eval_data.json \
  --max_workers 3
```

## Generation Evaluation

```bash
python3 -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --data_file ./evaluate_data/generation_eval_data.json \
  --output_file ./evaluate_result/generation_eval_result.json \
  --num_samples 3
```

**Note:** `Retrieval Evaluation` can be run independently. `Generation Evaluation` requires the output from `Answer Generation`, so `Answer Generation` must be executed first.
