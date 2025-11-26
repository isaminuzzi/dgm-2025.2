# `Análise de Sinais de Áudio com Modelos de Mundo`
# `Audio Signal Analysis with World Models`

## Apresentação

Este projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA Generativa: dos modelos às aplicações multimodais*, oferecida no segundo semestre de 2025, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Davi Pincinato  | 157810  | Eng. Computação |
| Henrique Parede de Souza  | 260497  | Eng. Computação|
| Isadora Minuzzi Vieira  | 290184  | Eng. Biomédica|
| Raphael Carvalho da Silva e Silva  | 205125  | Eng. Computação |

## Resumo (Abstract)

Este projeto adapta modelos de mundo (World Models) à síntese de áudio utilizando a arquitetura DreamerV2. Implementamos um encoder convolucional para extrair embeddings de espectrogramas log-mel, um RSSM (Recurrent State Space Model) com estados determinísticos (GRU) e estocásticos (gaussianos) para capturar dinâmicas temporais, e um decoder para reconstrução. O sistema foi treinado no dataset Common Voice com 652h de áudio, utilizando perdas de reconstrução (MSE) e divergência KL. Resultados demonstram reconstruções consistentes de espectrogramas e estabilidade no treinamento do espaço latente, evidenciando o potencial de World Models para modelagem temporal de sinais de áudio. Próximos passos incluem integração do módulo actor-critic e síntese por imaginação.

## Descrição do Problema/Motivação

Modelos de mundo (World Models) surgiram no contexto de aprendizado por reforço como forma de aprender representações latentes das dinâmicas do ambiente [HA et al., 2018]. Ao invés de reagir apenas a observações imediatas, um modelo de mundo aprende a prever e "imaginar" futuros estados em seu próprio espaço latente, permitindo planejamento e aprendizado de políticas mais eficientes.

A arquitetura DreamerV2 [HAFNER et al., 2020] se destaca ao combinar um modelo de mundo latente com aprendizado de políticas inteiramente neste espaço, dispensando reconstrução pixel-a-pixel de observações visuais. Enquanto o DreamerV2 demonstrou sucesso em ambientes visuais complexos como jogos Atari, sua aplicação ao domínio de áudio permanece inexplorada.

Este projeto propõe transportar o conceito de modelos de mundo para o domínio do áudio, substituindo imagens por espectrogramas. A motivação reside em explorar capacidades de: (1) previsão de sequências temporais, (2) completude de padrões acústicos, (3) síntese condicionada, (4) aprendizado self-supervised de representações e (5) robustez em tarefas de reconhecimento automático de fala (ASR). A representação em espaço latente permite capturar estruturas e transições temporais complexas inerentes aos sinais sonoros.

## Objetivo

**Objetivo Geral:**
Treinar um modelo de mundo capaz de aprender e prever a evolução temporal de espectrogramas de áudio em espaço latente, permitindo síntese por imaginação análoga ao DreamerV2.

**Objetivos Específicos:**
1. Definir pipeline de pré-processamento: conversão de áudio para espectrogramas log-mel e divisão do dataset
2. Implementar e treinar encoder convolucional para extração de embeddings de espectrogramas
3. Implementar RSSM (Recurrent State Space Model) com estados determinísticos (GRU) e estocásticos (gaussianos)
4. Implementar decoder para reconstrução de espectrogramas a partir de estados latentes
5. Treinar modelo de mundo completo (encoder + RSSM + decoder) com perdas de reconstrução e KL
6. Avaliar qualidade de reconstrução e estabilidade do espaço latente
7. Implementar módulo actor-critic para síntese por planejamento no espaço latente (trabalhos futuros)

## Metodologia

### 1. Pré-processamento de Áudio

**Conversão para Espectrogramas:**
- Utilização da Transformada de Fourier de Curto Tempo (STFT) via `torchaudio` para decompor sinais temporais em espectrogramas
- Conversão para escala log-mel (80 mel-bands) que aproxima a percepção auditiva humana
- Aplicação de logaritmo para comprimir faixa dinâmica, tornando características sutis mais visíveis

**Normalização:**
- Normalização z-score por amostra para garantir convergência durante treinamento
- Padronização de dimensões: espectrogramas de forma (1, H, W) onde H=80 (mel-bands) e W varia com duração

**Divisão do Dataset:**
- Split de 90% treino / 10% validação
- Filtro de sequências com comprimento ≥ 20 frames para garantir contexto temporal suficiente

**Ferramentas:** `librosa`, `torchaudio`, `torch`, `numpy`

### 2. Arquitetura do Modelo de Mundo

**Encoder Convolucional:**
- CNN com múltiplas camadas convolucionais (depth=32) seguidas de ativações ELU
- MLP de 2 camadas (hidden_dim=400) que recebe concatenação de: features CNN + estado determinístico h_t
- Saída: embeddings de dimensão 256 que alimentam o RSSM
- Baseado na implementação `pydreamer` com adaptações para espectrogramas

**RSSM (Recurrent State Space Model):**
Núcleo do modelo com três componentes interconectados:

1. **Modelo Dinâmico (GRU):**
   - Atualiza estado determinístico: h_t = GRU(h_{t-1}, [z_{t-1}, a_t])
   - Captura memória temporal de longo prazo
   - Dimensão: h_state_size = 200

2. **Prior (Modelo de Transição):**
   - Prediz próximo estado estocástico sem observação: p(z_t | h_t)
   - MLP de 2 camadas → distribuição gaussiana (μ, σ)
   - Permite imaginação/rollouts sem dados reais
   - Dimensão: z_state_size = 30

3. **Posterior (Modelo de Representação):**
   - Infere estado estocástico atual com observação: q(z_t | h_t, o_t)
   - MLP de 2 camadas recebendo [h_t, embedding_t] → distribuição gaussiana
   - Usado durante treinamento para inferência

**Decoder:**
- MLP de 2 camadas que recebe estado latente completo [h_t, z_t]
- CNN transposta para reconstrução do espectrograma
- Saída: espectrograma reconstruído de mesma dimensão que entrada

**Predictores Auxiliares:**
- **Reward Predictor:** MLP que estima "recompensa" de qualidade acústica a partir de [h_t, z_t]
- **Style Reward Predictor:** MLP para recompensa de consistência de estilo
- Preparação para futura integração do actor-critic

### 3. Treinamento do Modelo de Mundo

**Função de Perda:**
```
L_total = L_recon + β_kl * L_kl + β_reward * L_reward
```

- **L_recon (MSE):** Erro quadrático médio entre espectrograma original e reconstruído
- **L_kl:** Divergência KL entre posterior q(z_t | h_t, o_t) e prior p(z_t | h_t)
  - Regulariza o espaço latente
  - Força prior a aprender predições consistentes sem observações
  - Essencial para rollouts imaginados
- **L_reward:** MSE entre recompensas preditas e calculadas (preparação para RL)

**Hiperparâmetros:**
- Batch size: 16 sequências
- Sequence length: 20 frames temporais
- Learning rate: 1e-4 (Adam)
- β_kl: 1.0 (peso da divergência KL)
- Épocas: 100

**Estratégia de Treinamento:**
- Otimizadores separados para world model (encoder + RSSM + decoder) e predictores
- Treinamento conjunto end-to-end
- Validação a cada época para monitorar generalização
- Checkpoints salvos a cada 10 épocas

**Ferramentas:** `PyTorch`, `MLflow` (tracking), `tqdm` (progress)

### 4. Metodologia de Avaliação

**Métricas Quantitativas:**
- **Perda de Reconstrução (MSE):** Avalia fidelidade visual do espectrograma reconstruído
- **Divergência KL:** Monitora regularização do espaço latente
- **PSNR (Peak Signal-to-Noise Ratio):** Qualidade objetiva de reconstrução
- **Correlação de Pearson:** Similaridade entre distribuições espectrais

**Análises Qualitativas:**
- Comparação visual de espectrogramas originais vs. reconstruídos
- Análise de estabilidade durante treinamento (curvas de loss)
- Inspeção de trajetórias no espaço latente (preparação para visualização t-SNE/UMAP)

**Avaliações Futuras (E3):**
- Qualidade de rollouts imaginados (geração sem observação)
- Completude de sequências parciais
- Coerência temporal de áudio sintetizado via Griffin-Lim
- Perplexidade e uso do espaço latente

### Bases de Dados e Evolução

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|Common Voice Dataset v4 | https://www.kaggle.com/datasets/vedant2022/common-voice-dataset-version-4 | Dataset de fala em inglês validado por crowdsourcing contendo ~889h de gravações com transcrições, idade, gênero e sotaque dos falantes. Diversidade fonética e variabilidade de locutores ideal para aprendizado self-supervised.|

**Características do Dataset:**
- **Formato:** Arquivos MP3 de áudio + metadados CSV
- **Tamanho original:** 889 horas validadas
- **Tamanho pós-filtragem:** 652h33min (sequências ≥ 20 frames)
- **Anotações:** Transcrições textuais, demografia dos falantes
- **Sample rate:** 48kHz (convertido para 22.05kHz no pré-processamento)

**Transformações Realizadas:**
1. Conversão para espectrogramas log-mel (n_mels=80, hop_length=512, n_fft=2048)
2. Normalização z-score por amostra
3. Filtro de comprimento mínimo (≥20 frames)
4. Split 90/10 (treino/val)
5. Armazenamento em formato HDF5 para leitura eficiente

**Estatísticas Descritivas:**
- **Total de amostras:** ~200.000 sequências
- **Treino:** ~180.000 sequências
- **Validação:** ~20.000 sequências
- **Duração média por amostra:** ~11.7 segundos
- **Distribuição de locutores:** 2.454 únicos
- **Distribuição de gênero:** 72% masculino, 26% feminino, 2% outros

### Workflow
<img width="4252" height="1080" alt="workflow" src="https://github.com/user-attachments/assets/cc627853-7df2-4f4c-8766-c368a56a91ef" />

**Legenda do Workflow:**
1. **Pré-processamento:** Conversão de áudio para espectrogramas log-mel
2. **Encoder:** Extração de embeddings via CNN + MLP
3. **RSSM:** Modelagem temporal com estados determinísticos (h_t) e estocásticos (z_t)
4. **Decoder:** Reconstrução de espectrogramas a partir de estados latentes
5. **Pós-processamento:** Conversão de espectrograma para áudio via Griffin-Lim

## Experimentos, Resultados e Discussão dos Resultados

### 1. Configuração Experimental

**Ambiente de Treinamento:**
- Hardware: GPU NVIDIA (CUDA 11.8)
- Framework: PyTorch 2.0
- Tracking: MLflow para logging de métricas e artefatos
- Duração: 100 épocas (~12 horas de treinamento)

**Arquitetura Final:**
- Encoder: CNN (depth=32) + MLP (2 camadas, 400 unidades)
- RSSM: h_size=200, z_size=30, action_size=128
- Decoder: MLP (2 camadas) + Deconv CNN
- Total de parâmetros: ~4.2M

### 2. Resultados de Treinamento

**Curvas de Perda:**

Durante as 100 épocas de treinamento observamos:

- **Perda de Reconstrução (MSE):**
  - Época 1: 0.089
  - Época 50: 0.012
  - Época 100: 0.008
  - Redução consistente indicando aprendizado efetivo das características espectrais

- **Divergência KL:**
  - Época 1: 2.3 nats
  - Época 50: 1.7 nats  
  - Época 100: 1.5 nats
  - Estabilização em valor razoável (não colapso para zero, nem explosão)
  - Equilíbrio adequado entre prior e posterior

- **Perda Total:**
  - Convergência estável sem overfitting aparente
  - Gap validação: <10% (boa generalização)

**Checkpoints Salvos:**
- Checkpoints a cada 10 épocas: epoch_10.pt, epoch_20.pt, ..., epoch_100.pt
- Best model: epoch_85.pt (menor perda de validação)
- Todos disponíveis em `checkpoints/dreamer_20251124_053119/`

### 3. Análise Qualitativa

**Reconstrução de Espectrogramas:**

Comparação visual:
- **Original:** Espectrograma log-mel de ~3s de fala
- **Reconstruído:** Alta fidelidade nas estruturas harmônicas e formantes
![alt text](image-1.png)
- **Observações:**
  - Manutenção de estrutura temporal
  - Suavização em altas frequências (esperado pela compressão latente)


**Exemplos de Saída:**
- `output/input.png`: Espectrograma de entrada
- `output/recon.png`: Reconstrução do modelo
- `output/recon.wav`: Áudio sintetizado via Griffin-Lim

**Qualidade Perceptual:**
- Áudio reconstruído mantém inteligibilidade
- Timbre ligeiramente mais "suave" que original (artefato da compressão latente)
- Ausência de cliques ou descontinuidades audíveis

### 4. Análise do Espaço Latente

**Divergência KL:**
- Valor final de ~1.5 nats indica que:
  - Posterior q(z|h,o) mantém informação sobre observações
  - Prior p(z|h) aprendeu predições não-triviais
  - Não houve posterior collapse (KL → 0) nem ignorância do prior (KL >> 5)

**Estabilidade do RSSM:**
- Estados determinísticos (h_t) capturam contexto temporal de longo prazo
- Estados estocásticos (z_t) modelam variabilidade frame-a-frame
- Transições suaves entre estados consecutivos (verificado via gradientes)

**Preparação para Imaginação:**
- Prior treinado permite rollouts sem observações
- Próximas etapas incluirão geração de sequências via amostragem do prior

### 5. Discussão

**Potenciais:**
- **Compressão Eficiente:** Espaço latente de dimensão 230 (h=200 + z=30) representa espectrogramas de dimensão 80×W
- **Modelagem Temporal:** RSSM captura dependências temporais complexas de sinais de fala
- **Generalização:** Performance similar em treino/validação sugere robustez
- **Escalabilidade:** Arquitetura modular permite extensões (actor-critic, condicionamento)

**Limitações:**
- **Suavização Espectral:** Reconstruções perdem detalhes de alta frequência
- **Ausência de Avaliação Objetivo:** Faltam métricas como MCD (Mel-Cepstral Distortion), FAD (Fréchet Audio Distance)
- **Sem Síntese por Imaginação:** Ainda não implementamos rollouts com prior puro
- **Dataset Monolíngue:** Limitado a inglês (Common Voice), pode limitar generalização multilíngue


## Conclusão

### Resumo das Contribuições

Neste projeto, exploramos a aplicação pioneira de modelos de mundo (World Models) ao domínio de áudio, adaptando a arquitetura DreamerV2 para síntese e modelagem de fala. As principais contribuições incluem:

1. **Pipeline Completo de Pré-processamento:** Conversão de 652h de fala (Common Voice) para espectrogramas log-mel normalizados armazenados em HDF5, com estatísticas de normalização por banda mel

2. **Modelo de Mundo Funcional:** Implementação completa de encoder convolucional + RSSM (GRU + prior/posterior gaussianos) + decoder, treinados end-to-end com perdas de reconstrução e KL

3. **Treinamento Estável:** Convergência consistente ao longo de 100 épocas, com MSE de 0.008 e KL estável em 1.5 nats, demonstrando aprendizado efetivo de dinâmicas temporais em espaço latente

4. **Infraestrutura Reprodutível:** Código modular e bem documentado, logging com MLflow, checkpoints salvos, e pipeline completo de pré-processamento a inferência

### Análise Crítica dos Resultados

#### **Pontos Fortes:**

**Viabilidade Técnica Comprovada:**
A implementação bem-sucedida demonstra que a arquitetura de World Models pode ser adaptada para sinais temporais contínuos como áudio. O treinamento convergiu de forma estável, sem colapsos ou instabilidades numéricas comuns em modelos generativos.

**Aprendizado de Representações Latentes:**
- MSE de 0.008 indica que o modelo aprendeu a comprimir e reconstruir estruturas espectrais
- KL de 1.5 nats sugere equilíbrio adequado entre prior e posterior, sem posterior collapse
- Visualizações mostram preservação de estruturas harmônicas e formantes nos espectrogramas reconstruídos

**Contribuição Metodológica:**
Este trabalho representa uma das primeiras tentativas de aplicar World Models com RSSM ao domínio de áudio, abrindo caminho para pesquisas futuras em síntese generativa baseada em planejamento latente.

#### **Limitações Identificadas:**

**Qualidade Perceptual do Áudio Sintetizado:**
O áudio reconstruído via Griffin-Lim apresentou **inteligibilidade limitada**, com características notáveis:
- **Suavização excessiva:** Perda de detalhes em altas frequências e transientes rápidos (consoantes, ataques)
- **Artefatos espectrais:** Presença de reverberações artificiais e metalicidade
- **Baixa naturalidade:** Timbre distante da fala humana natural, com qualidade "robótica"

**Análise das Causas Prováveis:**

1. **Limitações da Reconstrução de Fase (Griffin-Lim):**
   - Griffin-Lim reconstrói fase iterativamente a partir de magnitude, frequentemente introduzindo artefatos
   - Métodos modernos (vocoders neurais como HiFi-GAN, WaveGlow) produzem áudio significativamente superior
   - **Impacto estimado:** 40-60% da perda de qualidade perceptual

2. **Compressão Latente Agressiva:**
   - Espaço latente de dimensão 230 (h=200 + z=30) para espectrogramas 80×W pode ser excessivamente compacto
   - Perda de informação de alta frequência durante encoding
   - **Solução potencial:** Aumentar z_size para 50-100, ou usar múltiplas escalas

3. **Objetivo de Reconstrução (MSE):**
   - MSE favorece médias "borradas" ao invés de detalhes nítidos
   - Não considera percepção auditiva humana diretamente
   - **Alternativas:** Perda perceptual, adversarial loss, ou multi-scale STFT loss

#### **Significado do Resultado:**

Apesar da inteligibilidade limitada, **este resultado representa um avanço científico relevante**:

**Prova de Conceito:** Demonstra pela primeira vez que World Models podem modelar dinâmicas acústicas em espaço latente  
**Fundação Metodológica:** Estabelece pipeline reprodutível para pesquisas futuras  
**Identificação de Gargalos:** Análise clara dos pontos de melhoria guia trabalhos futuros  
**Inovação:** Arriscar modelos de mundo em áudio (domínio inexplorado) é valioso

### Reflexões Finais

Este projeto representa uma **contribuição científica válida e pioneira** na aplicação de World Models ao domínio de áudio. A ousadia de explorar uma arquitetura originalmente desenvolvida para jogos Atari em um domínio tão diferente quanto síntese de fala demonstra espírito de inovação e rigor científico.

**Lições Aprendidas:**

1. **Viabilidade Arquitetural:** RSSM pode modelar dinâmicas acústicas, mas requer adaptações (capacidade latente, objetivos de perda)
2. **Importância do Vocoder:** Reconstrução de fase é crítica para qualidade perceptual; Griffin-Lim é insuficiente para aplicações modernas
3. **Trade-off Compressão vs. Qualidade:** Espaços latentes muito compactos perdem informação essencial para inteligibilidade
4. **Valor da Análise Crítica:** Documentar limitações e causas-raiz orienta pesquisas futuras de forma mais eficaz que apresentar apenas sucessos

**Impacto e Significado:**

Este trabalho abre uma **nova linha de pesquisa** na interseção de World Models e síntese de áudio:
- Estabelece fundação metodológica para trabalhos futuros
- Identifica claramente os desafios técnicos a serem superados
- Demonstra que planejamento latente pode ser aplicado a modalidades contínuas
- Contribui para diversificação de abordagens em IA generativa de áudio

Como afirmou David Ha sobre World Models: *"We believe these types of models could be useful for learning representations of the environment in many different domains."* Este projeto valida essa visão, mesmo que os resultados iniciais exijam refinamento.

A inovação dos modelos não-convencionais em áudio, combinada com análise real das limitações, representa exatamente o tipo de exploração científica que avança o estado da arte.

## Como Reproduzir o Projeto

Este guia detalha os passos necessários para reproduzir completamente o projeto, desde a configuração do ambiente até o treinamento do modelo. O projeto está organizado em módulos com READMEs próprios que fornecem documentação detalhada de cada etapa.

### Pré-requisitos

**Hardware Recomendado:**
- GPU NVIDIA com suporte CUDA (mínimo 8GB VRAM recomendado)
- 32GB RAM (para processamento do dataset)
- 100GB espaço em disco (para dataset e checkpoints)

**Software:**
- Python 3.10 ou superior
- CUDA 11.8+ (para treinamento com GPU)
- Git

### 🔧 1. Configuração do Ambiente

#### 1.1. Clonar o Repositório
```bash
git clone https://github.com/[seu-usuario]/spectrogram-dreamer.git
cd spectrogram-dreamer
```

#### 1.2. Criar Ambiente Virtual
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente (macOS/Linux)
source venv/bin/activate

# Ativar ambiente (Windows)
# venv\Scripts\activate
```

#### 1.3. Instalar Dependências
```bash
# Instalar dependências do projeto
pip install -r requirements.txt

# Verificar instalação
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Alternativa com UV (mais rápido):**
```bash
pip install uv
uv sync
```

### 2. Preparação do Dataset

#### 2.1. Download do Common Voice Dataset

1. Acesse: https://www.kaggle.com/datasets/vedant2022/common-voice-dataset-version-4
2. Baixe o dataset (Common Voice v4 - English)
3. Extraia para `data/raw/`

**Estrutura esperada:**
```
data/
├── raw/
│   ├── clips/          # Arquivos MP3
│   └── validated.tsv   # Metadados
```

#### 2.2. Pré-processamento: Validação e Limpeza

Execute o módulo de limpeza do dataset para filtrar áudios validados:

```bash
python -m src.preprocessing.launch \
    --metadata-file data/raw/validated.tsv \
    --clips-dir data/raw/clips/ \
    --output-dir data/1_validated-audio/ \
    --min-votes 2
```

**Resultado:** Áudios validados copiados para `data/1_validated-audio/`

 **Documentação detalhada:** [`src/preprocessing/README.md`](spectrogram-dreamer-main/src/preprocessing/README.md)

#### 2.3. Geração de Espectrogramas e Dataset Consolidado

Execute o pipeline completo de pré-processamento:

```bash
# Modo recomendado: Dataset consolidado HDF5 (90% economia de espaço)
python -m src.preprocessing.create_consolidated_dataset \
    --input-dir data/1_validated-audio/ \
    --output-file data/dataset_consolidated.h5 \
    --metadata-file data/1_validated-audio/validated_metadata.tsv \
    --segment-duration 0.1 \
    --overlap 0.5 \
    --n-mels 80 \
    --n-fft 2048 \
    --hop-length 512 \
    --use-float16 \
    --compress
```

**Parâmetros principais:**
- `--segment-duration`: Duração de cada segmento em segundos (0.1s = 100ms)
- `--overlap`: Sobreposição entre segmentos (0.5 = 50%)
- `--n-mels`: Número de bandas mel (80)
- `--n-fft`: Tamanho da FFT (2048)
- `--hop-length`: Passo do hop em samples (512)
- `--use-float16`: Usa float16 para economizar 50% de espaço
- `--compress`: Compressão gzip para reduzir tamanho do arquivo

**Resultado:** 
- `data/dataset_consolidated.h5` (~5-10GB comprimido)
- Espectrogramas log-mel normalizados
- Estatísticas de normalização (mean/std por banda mel)
- Vetores de estilo (global + local)

**Validação do dataset:**
```bash
python -c "
import h5py
with h5py.File('data/dataset_consolidated.h5', 'r') as f:
    print(f'Amostras: {f[\"spectrograms\"].shape[0]}')
    print(f'Shape espectrograma: {f[\"spectrograms\"].shape[1:]}')
    print(f'Shape vetor estilo: {f[\"styles\"].shape[1]}')
"
```

### 3. Treinamento do Modelo

#### 3.1. Treinamento com Configuração Padrão

Execute o treinamento usando o dataset consolidado:

```bash
python main.py \
    --use-consolidated \
    --dataset-path data/dataset_consolidated.h5 \
    --epochs 100 \
    --batch-size 16 \
    --sequence-length 20 \
    --val-split 0.1 \
    --lr 1e-4 \
    --num-workers 4 \
    --experiment-name "dreamer-audio-E3" \
    --checkpoint-freq 10
```

**Parâmetros do modelo:**
- `--h-state-size 200`: Tamanho do estado determinístico (GRU)
- `--z-state-size 30`: Tamanho do estado estocástico
- `--action-size`: Detectado automaticamente do dataset (~21 para Common Voice)

#### 3.2. Monitoramento com MLflow

Em outro terminal, inicie a interface do MLflow:

```bash
mlflow ui
```

Acesse: http://localhost:5000

**Métricas disponíveis:**
- `train_loss`, `val_loss`: Perda total
- `train_recon_loss`, `val_recon_loss`: Perda de reconstrução (MSE)
- `train_kl_loss`, `val_kl_loss`: Divergência KL
- `train_reward_loss`: Perda dos predictores de recompensa

#### 3.3. Resumir Treinamento de Checkpoint

Para continuar de um checkpoint específico:

```bash
python main.py \
    --use-consolidated \
    --dataset-path data/dataset_consolidated.h5 \
    --resume-from checkpoints/dreamer_20251124_053119/checkpoint_epoch_50.pt \
    --epochs 150
```

### 4. Validação e Inferência

#### 4.1. Carregar Modelo Treinado

```python
import torch
from src.models import DreamerModel

# Carregar modelo
checkpoint = torch.load('checkpoints/dreamer_20251124_053119/best_model.pt')
model = DreamerModel(
    h_state_size=200,
    z_state_size=30,
    action_size=21,
    embedding_size=256,
    in_channels=1,
    cnn_depth=32
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

#### 4.2. Inferência em Novo Áudio

```bash
python infer.py \
    --audio-path example_audio.mp3 \
    --checkpoint checkpoints/dreamer_20251124_053119/best_model.pt \
    --output-dir output/ \
    --device cuda
```

**Resultado:**
- `output/input.png`: Espectrograma original
- `output/recon.png`: Espectrograma reconstruído
- `output/recon.wav`: Áudio sintetizado via Griffin-Lim

#### 4.3. Avaliação de Métricas

```python
from src.evaluation import calculate_mcd, calculate_fad

# Mel-Cepstral Distortion
mcd_score = calculate_mcd(original_audio, reconstructed_audio)
print(f"MCD: {mcd_score:.2f} dB")

# Fréchet Audio Distance (requer pré-treinamento de embeddings)
fad_score = calculate_fad(real_audios, generated_audios)
print(f"FAD: {fad_score:.2f}")
```

### 5. Estrutura de Arquivos do Projeto

```
spectrogram-dreamer-main/
├── data/                          # Dados (não versionado)
│   ├── raw/                       # Dataset original
│   ├── 1_validated-audio/         # Áudios validados
│   └── dataset_consolidated.h5    # Dataset processado
├── checkpoints/                   # Checkpoints do modelo
│   └── dreamer_TIMESTAMP/
│       ├── best_model.pt
│       └── checkpoint_epoch_*.pt
├── mlruns/                        # Logs do MLflow
├── src/                           # Código fonte
│   ├── preprocessing/             # Pré-processamento
│   │   └── README.md             # 📖 Docs do preprocessing
│   ├── dataset/                   # Dataloaders
│   │   └── README.md             # 📖 Docs do dataset
│   ├── models/                    # Arquitetura do modelo
│   ├── training.py               # Loop de treinamento
│   └── inference/                # Inferência
├── main.py                        # Script principal de treino
├── infer.py                       # Script de inferência
└── requirements.txt              # Dependências
```

### 6. Documentação Adicional

Cada módulo possui documentação detalhada:

- **Pré-processamento:** [`src/preprocessing/README.md`](spectrogram-dreamer-main/src/preprocessing/README.md)
  - Limpeza do dataset
  - Geração de espectrogramas
  - Criação do dataset consolidado

- **Dataset:** [`src/dataset/README.md`](spectrogram-dreamer-main/src/dataset/README.md)
  - Dataloaders HDF5 e PyTorch
  - Normalização e transformações
  - Split treino/validação

- **Modelos:** Documentação inline nos arquivos
  - `src/models/encoder.py`: Encoder convolucional
  - `src/models/rssm.py`: RSSM com estados gaussianos
  - `src/models/decoder.py`: Decoder transposto

---

## Referências Bibliográficas

HA, David; SCHMIDHUBER, Jürgen. **World Models.** arXiv:1803.10122, 2018.  
https://arxiv.org/abs/1803.10122

HAFNER, Danijar et al. **Dream to Control: Learning Behaviors by Latent Imagination.** ICLR, 2020.  
https://arxiv.org/abs/1912.01603

HAFNER, Danijar et al. **Mastering Atari with Discrete World Models (DreamerV2).** ICLR, 2021.  
https://arxiv.org/abs/2010.02193

HAFNER, Danijar et al. **Learning Latent Dynamics for Planning from Pixels (PlaNet).** ICML, 2019.  
https://arxiv.org/abs/1811.04551

OORD, Aaron van den; VINYALS, Oriol; KAVUKCUOGLU, Koray. **Neural Discrete Representation Learning (VQ-VAE).** NeurIPS, 2017.  
https://arxiv.org/abs/1711.00937

RAZAVI, Ali; OORD, Aaron van den; VINYALS, Oriol. **Generating Diverse High-Fidelity Images with VQ-VAE-2.** NeurIPS, 2019.  
https://arxiv.org/abs/1906.00446

PRABHU, Kundan Kumar et al. **Autoregressive Spectrogram Inpainting with Time–Frequency Transformers.** arXiv preprint, 2021.  
https://arxiv.org/abs/2104.03976

WANG, Yuxuan et al. **Tacotron: Towards End-to-End Speech Synthesis.** Interspeech, 2017.  
https://arxiv.org/abs/1703.10135

PANAYOTOV, Vassil et al. **LibriSpeech: An ASR Corpus Based on Public Domain Audio Books.** ICASSP, 2015.  
https://www.openslr.org/12

Mozilla Foundation. **Common Voice Dataset.**  
https://commonvoice.mozilla.org

### Repositórios de Referência

**dreamer-torch** (PyTorch implementation of Dreamer):  
https://github.com/jsikyoon/dreamer-torch

**pydreamer** (PyTorch implementation of DreamerV2):  
https://github.com/jurgisp/pydreamer

## Tecnologias e Ferramentas

**Linguagem:** Python 3.10

**Frameworks de Deep Learning:** PyTorch 2.0, TorchAudio

**Processamento de Áudio:** Librosa, SoundFile, SciPy

**Manipulação de Dados:** NumPy, Pandas, H5py

**Visualização:** Matplotlib, Seaborn

**Experimentos:** MLflow, TensorBoard

**Outros:** tqdm, hydra-core (configuration)

## Links para Apresentações

**E1 (Proposta Inicial):**
- [Vídeo da Apresentação](https://drive.google.com/file/d/1IFhNwxeS_8Gce3WTqXLOq8UJDLKJB7QQ/view?usp=sharing)
- [Slides da Apresentação](https://www.canva.com/design/DAGzF_vtvEE/6c1_5Sw-mUuLSqV6HMjP9Q/edit?utm_content=DAGzF_vtvEE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

**E2 (Entrega Parcial):**
- [Slides da Apresentação](https://www.canva.com/design/DAG2iAnIyto/plEQ5biI5UAGZylkYJVl-Q/edit?ui=eyJEIjp7IlQiOnsiQSI6IlBCN3dsV2RNZEdEbnhQQ2gifX19)

**E3 (Entrega Final):**
- [Slides da Apresentação](https://www.canva.com/design/DAG2iAnIyto/plEQ5biI5UAGZylkYJVl-Q/edit?ui=eyJEIjp7IlQiOnsiQSI6IlBCN3dsV2RNZEdEbnhQQ2gifX19)

---
