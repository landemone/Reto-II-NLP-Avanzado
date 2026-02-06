# LDA Report

- Generated: 2026-02-06T01:31:14Z
- Documents: 663
- Topics: 10

## Topics
### Topic 0 - Trabajo en Argentina
Top words: cada, estamos, argentina, aca, hacer, gracias, vamos, pais, va, trabajo, argentinos, ustedes

### Topic 1 - Educacion y futuro
Top words: chicos, educacion, estamos, ustedes, gracias, aca, docentes, cada, hoy, futuro, realmente, ahi

### Topic 2 - Cosas del mundo
Top words: entonces, ustedes, cosas, anos, hacer, aca, verdad, ahi, mundo, argentina, va, estamos

### Topic 3 - Pobreza y gobierno
Top words: estado, gobierno, pais, argentina, vamos, hoy, anos, argentinos, cada, va, pobreza, hacer

### Topic 4 - Trabajo y unidad argentina
Top words: argentinos, hoy, estamos, anos, pais, cada, gracias, quiero, juntos, argentina, ustedes, trabajo

### Topic 5 - Obras en la ciudad
Top words: estamos, anos, obra, argentinos, obras, haciendo, aca, pais, trabajo, hacer, hemos, ciudad

### Topic 6 - Salud y trabajo
Top words: anos, ustedes, trabajo, argentina, gracias, estamos, salud, aca, cada, va, aplausos, hoy

### Topic 7 - Energia y trabajo
Top words: estamos, trabajo, anos, mundo, argentinos, argentina, cada, pais, hemos, hoy, energia, vamos

### Topic 8 - Cambio climatico y energias
Top words: climatico, cambio climatico, hemos, va, estamos, renovables, cada, parque, energias, realmente, ambiente, pais

### Topic 9 - Argentina y presidente
Top words: argentina, presidente, mundo, hemos, paises, pais, ha, estamos, asi, gracias, quiero, muchas

## Config
```json
{
  "input_dir": "data/raw",
  "output_dir": "outputs",
  "preprocess": {
    "lowercase": true,
    "normalize_accents": true,
    "remove_numbers": true,
    "min_token_len": 2,
    "stopwords_path": null,
    "extra_stopwords": []
  },
  "vectorize": {
    "max_features": 5000,
    "min_df": 2,
    "max_df": 0.95,
    "ngram_range": [
      1,
      2
    ]
  },
  "lda": {
    "n_topics": 10,
    "max_iter": 20,
    "learning_method": "batch",
    "random_state": 42,
    "doc_topic_prior": null,
    "topic_word_prior": null
  },
  "cache": {
    "enabled": true,
    "dir": "outputs/cache"
  },
  "llm": {
    "enabled": true,
    "model": "gpt-4o-mini",
    "language": "es",
    "temperature": 0.2,
    "max_tokens": 200,
    "api_key": null,
    "base_url": null,
    "organization": null,
    "project": null
  },
  "report": {
    "top_words": 12,
    "report_path": "outputs/report.md"
  }
}
```