# CancerOral-CNNs
Repositório oficial do código utilizado no estudo que avalia o desempenho de arquiteturas convolucionais pré-treinadas (DenseNet121, ResNet18 e GoogLeNet) na detecção de câncer oral usando Transfer Learning e Validação Cruzada 5-Fold. O projeto implementa todo o pipeline de treinamento, avaliação, geração de métricas e matrizes de confusão.

O objetivo deste projeto é comparar o desempenho de três arquiteturas CNN pré-treinadas — **DenseNet121**, **ResNet18** e **GoogLeNet** — aplicadas à classificação binária de imagens orais (Câncer / Não Câncer) utilizando **Transfer Learning** e **Validação Cruzada 5-Fold**.

São gerados automaticamente:

* Gráficos de *Loss* e *Accuracy* por época
* Matrizes de confusão
* Curvas ROC/AUC
* Arquivos CSV contendo probabilidades e rótulos preditos
* Sumários de métricas por fold
* Versões finais dos pesos de cada fold

Todo o pipeline foi desenvolvido para garantir **reprodutibilidade**, podendo ser executado integralmente no **Google Colab**.

---

# 🗂 **Dataset Utilizado**

O dataset público foi disponibilizado por **Mohd Zaid Rashid** no Kaggle:

🔗 **Oral Cancer Dataset – Kaggle**
[https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset)

Licença: **Apache 2.0** → permite redistribuição, modificação e uso comercial, incluindo publicação em artigos.

### Estrutura das classes:

* `Câncer`: 500 imagens
* `Não Câncer`: 450 imagens
  Total: **950 imagens clínicas coloridas da cavidade oral**

As imagens apresentam variação natural de:

* Iluminação
* Foco
* Ângulo
* Região anatômica

O dataset **não contém confirmação histológica**, consistindo apenas em fotografias clínicas.

---

# ⚙️ **Requisitos**

O código foi projetado para rodar no **Google Colab**, mas pode ser executado localmente usando Python 3.8+.

### Bibliotecas principais:

```
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
kaggle
```

No Colab, as dependências já estão presentes (exceto `kaggle`).

---

# 🚀 **Como Executar**

## ✅ 1. Abra no Google Colab (recomendado)

Faça upload do arquivo `OralCancer.ipynb` ou execute diretamente no notebook.

## ✅ 2. Adicione sua API Key do Kaggle

Antes de rodar, faça upload do arquivo `kaggle.json`.

No Colab:

```python
from google.colab import files
files.upload()
```

## ✅ 3. Baixe o Dataset

O script já contém:

```bash
!kaggle datasets download -d zaidpy/oral-cancer-dataset
!unzip -q oral-cancer-dataset.zip -d data
```

## ✅ 4. Execute o treinamento

Rode o bloco no Colab.

---

# **Modelos Avaliados**

| Modelo          | Parametrização | Características                                         |
| --------------- | -------------- | ------------------------------------------------------- |
| **ResNet18**    | 11M params     | Camadas residuais simples, alto controle de overfitting |
| **DenseNet121** | 8M params      | Conexões densas, melhor fluxo de gradiente              |
| **GoogLeNet**   | 6.8M params    | Módulos Inception, baixo custo computacional            |

---

# **Metodologia: Validação Cruzada 5-Fold**

O código utiliza:

* `KFold(shuffle=True, random_state=42)`
* 80% treino / 20% validação por fold
* 5 execuções independentes para cada modelo
* Registro completo das métricas por fold

---

# 📊 **Outputs Gerados**

Para cada modelo e para cada fold, são gerados automaticamente:

### ✔️ Curvas de Loss

`{model}/fold_i/{model}_Loss.png`

### ✔️ Curvas de Accuracy

`{model}/fold_i/{model}_Accuracy.png`

### ✔️ Matriz de Confusão

`{model}/fold_i/{model}_confusion_matrix.png`

### ✔️ Curva ROC

`{model}/fold_i/{model}_roc_curve.png`

### ✔️ CSV com probabilidades

`{model}/fold_i/{model}_test.csv`

### ✔️ CSV com métricas por fold

`{model}/{model}_folds_summary.csv`

---

# 📈 **Resultados (Resumo do Artigo)**

| Modelo          | Recall    | Acurácia  | F1-score  | Precisão  |
| --------------- | --------- | --------- | --------- | --------- |
| **DenseNet121** | **0.945** | 0.937     | **0.951** | 0.957     |
| **GoogLeNet**   | 0.936     | **0.943** | 0.931     | 0.927     |
| **ResNet18**    | 0.935     | 0.936     | 0.947     | **0.960** |

A **DenseNet121** foi a arquitetura com melhor desempenho geral.

---

#  **Acknowledgments**

```
The authors would like to thank the Kaggle platform and Mohd Zaid Rashid for 
providing the publicly available Oral Cancer Dataset used in this study. 
We also acknowledge the computational support provided by Google Colab, 
which enabled the training and evaluation of the convolutional neural networks.
```
---

# **Licença**

O código deste repositório é disponibilizado sob a licença **MIT**.
O dataset é licenciado sob **Apache 2.0** e deve ser citado adequadamente.


Quer adicionar algo?

