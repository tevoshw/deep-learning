# Model Monitoring Dashboard

1. Monitoramento da memória
2. Monitoramento dos dados
3. Monitoramento do treino do modelo (ao vivo)
4. Monitoramento do modelo (estático (pós-treino e ao vivo))
5. Estudo do modelo já pronto

## 1. Memory & Infrastructur
### Basis
First thing to understand it's the basis of hardware, in the world of computing, both the CPU and the GPU are crucial processing units, but they were designed for completely different tasks. While one focuses on intelligence and command, the other focuses on brute force and repetition.

#### 1. O que é a CPU? (Central Processing Unit)
A CPU é o "cérebro" do computador. Ela é responsável por executar o sistema operacional e controlar todos os outros componentes. Sua principal característica é a versatilidade: ela consegue lidar com qualquer tipo de instrução, desde cálculos matemáticos até a lógica complexa de um software de edição ou um navegador web.

- Arquitetura: Poucos núcleos (cores), mas extremamente potentes e rápidos.
- Tipo de Processamento: Sequencial (uma tarefa por vez, em altíssima velocidade).
- Ponto forte: Baixa latência e tomada de decisão lógica complexa (instruções "se... então").

#### 2. O que é a GPU? (Graphics Processing Unit)
A GPU começou como um processador especializado apenas em renderizar gráficos de vídeo (daí o nome). No entanto, percebeu-se que sua arquitetura é perfeita para cálculos matemáticos massivos. Ela não tenta ser "inteligente" como a CPU, mas é incrivelmente eficiente em fazer a mesma coisa milhares de vezes ao mesmo tempo.

- Arquitetura: Milhares de núcleos (cores) pequenos e especializados.
- Tipo de Processamento: Paralelo (milhares de tarefas simples executadas simultaneamente).
- Ponto forte: Alta vazão (throughput) de dados e processamento de matrizes.

### 1.1. The RAM vs VRAM
Hierarquia de memória — entender por que VRAM > RAM > disco em termos de velocidade/custo te ajuda a tomar decisões de onde colocar dados, usar memory-mapped files, etc.
Bandwidth vs Latência — a diferença entre os dois explica por que operações matriciais são "memory-bound" em certos cenários e "compute-bound" em outros. Isso aparece direto em profiling com torch.profiler.
HBM vs GDDR vs DDR — os tipos de memória que aparecem nas specs de GPUs (H100, A100, RTX 4090). Saber a diferença te ajuda a interpretar benchmarks.
PCIe e NVLink — como CPU e GPU se comunicam, e por que mover tensores entre dispositivos é caro. Relevante quando você tem multi-GPU.
O que você pode pular:
Timing de RAM, overclocking, slots físicos — isso não agrega nada para ML/dados.
Sugestão de caminho:
Os artigos do blog da Hugging Face sobre "model memory anatomy" e as docs do PyTorch sobre CUDA memory management cobrem exatamente o que você precisa, já no contexto de ML — sem rodeios de hardware puro.

### 1.2 Dtypes & Memory Precision
1. Bit: 0 ou 1, permite sequência de até 8 bits
2. Byte: Uma sequência de 8 bits
- Tipos de dados (bit p sinal, bit pra expoente, bit para fração), sempre representam quantos bits ocupam na memória
* **Float16:** Suporta com segurança 3 a 4 casas decimais, ocupa 16bits/2bytes na memória
* **Float32:** Suporta com segurança 7 casas decimais, ocupa 32bits/4bytes na memória
* **Float64:** Suporta com segurança 15 a 17 casas decimais, ocupa 64bits/8bytes na memória
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
