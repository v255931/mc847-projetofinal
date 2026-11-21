# mc847-projetofinal

Trabalho final de Aprendizado Não Supervisionado (MC847, Unicamp, 2s2026): modelos de recomendação e análise de espaços latentes no MovieLens 100K.

Enunciado: https://www.ic.unicamp.br/mraimundo/teaching/unsupervised-learning/projeto.html

Relatório: [`relatorio.pdf`](relatorio.pdf)

## Conteúdo

| Caminho | O que é |
|---|---|
| `unsupervised_assignment_2s2026.ipynb` | Notebook com hipóteses, código, experimentos executados e discussões |
| `data/ml-100k/` | MovieLens 100K (já incluído no repositório) |
| `results/` | Tabelas (CSV) geradas pelo notebook |
| `figs/` | Figuras geradas pelo notebook |
| `pyproject.toml`, `uv.lock`, `.python-version` | Ambiente reprodutível (Python 3.12, versões travadas) |

## Reproduzir

Requisito: [`uv`](https://docs.astral.sh/uv/) instalado.

```bash
# 1. Criar o ambiente exatamente como em uv.lock (inclui o JupyterLab)
uv sync --locked

# 2. Abrir o Jupyter Lab
uv run jupyter lab
```

Abra `unsupervised_assignment_2s2026.ipynb` e use `Run All`. As figuras e tabelas são regravadas em `figs/` e `results/`.

Observações:

- **Dados:** `data/ml-100k/` já está no repositório e o notebook o usa direto. Só se a pasta não existir ele baixa o arquivo de `files.grouplens.org`.
- **Sementes:** tudo é fixado com `set_seed(42)` (Python, NumPy, PyTorch), e cada dobra da validação cruzada usa `42 + dobra`. Os números do relatório vieram de uma execução no Google Colab (CPU). Em outra máquina, diferenças de BLAS, de versão do PyTorch e de número de *threads* podem alterar os resultados na casa decimal; as conclusões não devem mudar.
- **Tempo:** a execução completa levou cerca de 17 minutos em CPU no Colab. As partes mais pesadas são o PPCA com `k=20` e os VAEs na validação cruzada.
- **PyTorch:** em Linux, `uv.lock` resolve o PyTorch com CUDA, o que baixa vários GB de dependências. Em máquina sem GPU funciona do mesmo jeito (usa CPU), só com download maior. O código escolhe `cuda` se houver GPU e `cpu` caso contrário.
- **Última célula:** gera `resultados_mc847.zip` (figuras e tabelas) e, no Colab, dispara o download. Fora do Colab, o arquivo fica na pasta do projeto (ignorado pelo git).

## Notas de implementação

- A fatoração de matrizes (SGD) e o EM do PPCA têm versões aceleradas (`numba` e operações matriciais). As classes de referência foram mantidas e o notebook verifica, com `assert`, que as versões rápidas reproduzem os mesmos parâmetros com a mesma semente.
- O VAE usa o código de referência em PyTorch, com posterior gaussiana diagonal por usuário.
