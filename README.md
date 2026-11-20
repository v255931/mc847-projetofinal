# mc847-projetofinal

Trabalho final de Aprendizado Não Supervisionado (MC847, Unicamp, 2s2026): modelos de recomendação e análise de espaços latentes no MovieLens 100K.

Enunciado: https://www.ic.unicamp.br/mraimundo/teaching/unsupervised-learning/projeto.html

## Reproduzir

```bash
# 1. Criar o ambiente (Python 3.12, versões travadas em uv.lock)
uv sync

# 2. Abrir o Jupyter Lab
uv run --with jupyter jupyter lab
```

Abra `unsupervised_assignment_2s2026.ipynb` e execute todas as células (`Run All`). O notebook baixa o MovieLens 100K para `data/` na primeira execução. As sementes são fixadas com `set_seed(42)`.

Figuras e tabelas são gravadas em `figs/` e `results/`.

