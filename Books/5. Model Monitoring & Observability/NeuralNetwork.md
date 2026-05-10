# Model Monitoring Dashboard

1. Monitoramento da memória
2. Monitoramento dos dados
3. Monitoramento do treino do modelo (ao vivo)
4. Monitoramento do modelo (estático (pós-treino e ao vivo))
5. Estudo do modelo já pronto

## 1. Memory & Infrastructur
Hierarquia de memória — entender por que VRAM > RAM > disco em termos de velocidade/custo te ajuda a tomar decisões de onde colocar dados, usar memory-mapped files, etc.
Bandwidth vs Latência — a diferença entre os dois explica por que operações matriciais são "memory-bound" em certos cenários e "compute-bound" em outros. Isso aparece direto em profiling com torch.profiler.
HBM vs GDDR vs DDR — os tipos de memória que aparecem nas specs de GPUs (H100, A100, RTX 4090). Saber a diferença te ajuda a interpretar benchmarks.
PCIe e NVLink — como CPU e GPU se comunicam, e por que mover tensores entre dispositivos é caro. Relevante quando você tem multi-GPU.
O que você pode pular:
Timing de RAM, overclocking, slots físicos — isso não agrega nada para ML/dados.
Sugestão de caminho:
Os artigos do blog da Hugging Face sobre "model memory anatomy" e as docs do PyTorch sobre CUDA memory management cobrem exatamente o que você precisa, já no contexto de ML — sem rodeios de hardware puro.
### 1.1 VRAM Usage (Allocated vs Reserved)
**VRAM (Video RAM) is the dedicated memory of the GPU, used to arm the data that the graphics processor needs to access quickly.**

### 1.2 Dtypes & Memory Precision
- Tipos de dados (float16, 32, 64)
- Impacto na VRAM
- Quantização (Reduzir tipo de dado 32 -> 16)
### 1.3 Peak VRAM Consumption
- Max que o VRAM aguenta antes de morrer

## 2. Data Flow & Speed
### 2.1 DataLoader Bottleneck Check
Os dados ficam na CPU, e precisamos levar para a GPU, podendo assim dar problemas, aqui medimos se a CPU está demorando para preparar o batch
(leitura do disco, preprocessing, augmentation)
- num_workers
- Velocidade do disco
### 2.2 Tokens/Samples per second (Throughput)
- Samples/Images/Tokens per second (analyzes)

### 2.3 Host-to-Device Latency
Mede se a transferência do batch pronto (RAM → VRAM) está lenta, mede o tempo dessa transferência
- PCIe
- pin_memory
- non_blocking

## 3. Training Dynamics (Live)
### 3.1 Loss (Train vs Validation)
### 3.2 Learning Rate Schedule
### 3.3 Gradient Norm (por camada)
### 3.4 Update-to-Weight Ratio
### 3.5 Activation Saturation
### 3.6 NaN / Infinity Detection

## 4. Structural Health (Static)
### 4.1 Parameters Count (Total vs Trainable)
### 4.2 Weight Distribution (Mean/Std/Min/Max)
- Entender a distribuição dos pesos
### 4.3 Weight Sparsity (%)
- Nem todos os pesos de um modelo contribuem igualmente. Alguns são tão pequenos que praticamente não influenciam o resultado — são "inúteis" na prática
### 4.4 Dead Neurons
- Um neurônio é composto por vários pesos. Um neurônio pode ter pesos não-zero mas ainda assim nunca ativar — porque a combinação deles sempre resulta em zero após a função de ativação.
### 4.5 Layer Force (L2 Norm)
- Treino rodando normalmente... Loss de repente vai para NaN, Você perdeu horas de treino sem saber o que aconteceu, aqui precisamos aonde aconteceu e o que
- Modelo treinado, accuracy ok...Mas 40% das camadas não contribuem em nada Você tem um modelo 40% maior do que precisava Pagou por VRAM e compute à toa
o.
### 4.6 Parameter Norm Ratio
- razão entre o tamanho dos pesos e dos gradientes
- indica se o learning rate está adequado

## 5. Interpretabilidade
