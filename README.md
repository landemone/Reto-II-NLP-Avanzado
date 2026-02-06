# Reto LDA ChatGPT

Pipeline completo para modelado de temas (LDA) y etiquetado con LLM.

## Requisitos
- Python 3.10+
- Dependencias en `pyproject.toml`

## Uso rapido
```bash
python -m src.run_pipeline
```

Opciones utiles:
```bash
python -m src.run_pipeline --input-dir data/raw --output-dir outputs --topics 12
python -m src.run_pipeline --no-llm
python -m src.run_pipeline --config config.json
```

## Configuracion (JSON)
Ejemplo minimo:
```json
{
  "input_dir": "data/raw",
  "output_dir": "outputs",
  "lda": { "n_topics": 10 }
}
```

## OpenAI (etiquetas con LLM)
Definir variable de entorno:
```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_LANG="es"
```

Si no hay API key o se usa `--no-llm`, el pipeline solo genera top words.

## Salidas
- `outputs/lda_model/model.pkl`
- `outputs/lda_model/vectorizer.pkl`
- `outputs/doc_topics.csv`
- `outputs/topics_top_words.csv`
- `outputs/topics_labels.csv` (si LLM)
- `outputs/llm_error.txt` (si fallo LLM o parseo)
- `outputs/report.md`
