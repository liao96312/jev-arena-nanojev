# Jev Arena × NanoJev：GTX 1660 Super 6GB 纯本地技术方案

> 目标：在一台 Windows + GTX 1660 Super 6GB 的电脑上，做一个 **完全本地、不调用任何外部推理 API** 的 Jev 决策游戏 Demo。
> 核心玩法：机器人在 20×20 Arena 中抢宝石、躲避火焰、攻击怪物、吃血包；NanoJev 根据结构化状态对候选动作一次性打分，并驱动游戏。
> 项目定位：不是“做一个小游戏”，而是做一个 **System-1 决策模型实验场**，可以比较 Random / Rule / NanoJev、多种策略、概率分布和长期收益。

---

## 0. 最终想做成什么

最终界面建议：

```text
┌──────────────────────────────────────────────┐
│                 JEV ARENA                    │
├────────────────────────┬─────────────────────┤
│                        │ NanoJev Decision    │
│   💎          👾       │                     │
│        ███             │ MOVE_E    51.2% ███│
│              🔥        │ ATTACK    27.4% ██ │
│     🤖                 │ MOVE_N    13.1% █  │
│                 ❤️     │ WAIT       8.3%    │
│                        │                     │
├────────────────────────┼─────────────────────┤
│ HP 63   Score 284      │ latency: 183 ms    │
│ Gems 12  Kills 4       │ entropy: 1.12      │
│ Tick 193 / 500         │ model: NanoJev     │
└────────────────────────┴─────────────────────┘
```

支持三种 Agent 一键切换：

```text
Random Agent
Rule Agent
NanoJev Agent
```

最终可以做到：

```text
同一个地图 seed
      │
 ┌────┼───────────────┐
 ↓    ↓               ↓
Random Rule        NanoJev
 ↓    ↓               ↓
跑100局统一评测
      │
      ↓
胜率 / 生存时间 / 宝石 / 击杀 / Reward / 延迟
```

第二阶段再扩展成：

```text
莽夫 NanoJev
苟王 NanoJev
贪财 NanoJev
```

让不同策略模型在同一个 Arena 中表现出不同“性格”。

---

# 1. 为什么选择 NanoJev

NanoJev 的接口和这个项目天然匹配：

```text
State
 +
Question
 +
Dynamic Candidates
 ↓
NanoJev
 ↓
完整动作概率分布
```

而不是传统 LLM 的：

```text
Prompt
 ↓
逐 Token 生成一段文字
 ↓
解析文字
 ↓
执行动作
```

NanoJev 当前基于 Qwen3-0.6B，并提供：

- Choice：动态候选动作；
- Boolean：真假判断；
- Score：等级评分；
- 多状态并行；
- 多问题并行；
- 单次前向返回完整概率分布；
- 本地持久推理服务；
- 迷宫 / 贪吃蛇训练和评测流程。

本项目第一阶段只使用 `Choice`。

参考：

- NanoJev: https://github.com/TianyuCodings/NanoJev
- 中文 README: https://github.com/TianyuCodings/NanoJev/blob/main/README.zh-CN.md
- Pipeline Runbook: https://github.com/TianyuCodings/NanoJev/blob/main/research/pipeline_runbook.md

---

# 2. GTX 1660 Super 的现实约束

你的 GTX 1660 Super：

```text
Architecture: Turing
CUDA Cores: 1408
VRAM: 6GB GDDR6
Tensor Core: 无
```

因此方案必须围绕 **6GB VRAM** 设计。

## 2.1 不能直接照搬 NanoJev 官方训练参数

NanoJev 当前官方示例主要使用：

```text
--precision bf16
microbatch_questions = 4
max_length = 512
完整 Qwen3-0.6B backbone 参与训练
```

而官方 requirements 中注明其实际开发环境是 A100 80GB。

1660S 不适合照搬。

我们的原则：

```text
推理              → FP32 先跑通
训练              → 第一版只训练决策头
输入长度          → 128~256 tokens
microbatch        → 1 question
梯度累积          → 保留较大的 effective batch
完整 Backbone 微调 → 第一阶段禁止
```

### 为什么不第一天就全量微调

Qwen3-0.6B：

```text
FP32 权重大约：
0.6B × 4 byte ≈ 2.4GB
```

全参数训练还需要：

```text
模型权重
梯度
Adam 一阶动量
Adam 二阶动量
Activation
CUDA Workspace
```

6GB 显存很容易直接 OOM。

所以：

> **1660S 第一阶段目标不是 Full Fine-tuning，而是 Frozen Backbone + Decision Head。**

等整个项目跑通以后，再实验：

```text
Head only
↓
Head + 最后一层 Transformer
↓
Head + 最后两层
↓
LoRA / 其他 PEFT
```

而不是反过来。

---

# 3. 总体系统架构

```text
                        ┌─────────────────┐
                        │   Pygame UI     │
                        │    60 FPS       │
                        └────────┬────────┘
                                 │
                                 ↓
┌────────────────────────────────────────────────────┐
│                    ArenaEnv                        │
│                                                    │
│ map / player / enemy / gem / fire / medkit        │
│ collision / reward / combat / terminal             │
└───────────────┬─────────────────────────┬──────────┘
                │                         │
                ↓                         ↓
       StateEncoder              CandidateBuilder
                │                         │
                └─────────────┬───────────┘
                              ↓
                    Decision Request
                              │
         ┌────────────────────┼────────────────────┐
         ↓                    ↓                    ↓
   RandomAgent           RuleAgent          NanoJevAgent
                                                   │
                                                   ↓
                                          Local NanoJev
                                          localhost:8765
                                                   │
                                                   ↓
                                           probability[]
                              │
                              ↓
                         PolicySelector
                              │
                     greedy / sampling
                              │
                              ↓
                          env.step()
                              │
                              ↓
                          ReplayLogger
```

注意：

`localhost:8765` 只是本机进程通信，不是外部 API。

整个运行链路：

```text
你的游戏
↓
127.0.0.1
↓
你的 NanoJev
↓
你的 1660S
```

完全可以断网运行。

---

# 4. 第一版游戏设计

项目名暂定：

```text
Jev Arena
```

## 4.1 地图

建议：

```text
20 × 20 Grid
```

第一版不要做连续物理坐标。

原因：

- 数据更容易生成；
- 状态更容易表达；
- 行为更容易解释；
- 可以确定性 replay；
- 可以做 rollout；
- 可以快速批量生成几十万状态；
- 更适合验证决策模型。

---

## 4.2 地图元素

```text
.  空地
#  墙
P  Player
E  Enemy
G  Gem
F  Fire
H  Health Pack
```

第一版数量：

```text
Player       1
Enemy        2~5
Gem          3~8
Fire         5~15
Health Pack  0~3
Wall         随机生成
```

---

# 5. 动作空间

不要固定所有动作永远都能选。

使用 NanoJev 的 Dynamic Candidates。

完整动作集合：

```text
MOVE_N
MOVE_S
MOVE_W
MOVE_E

ATTACK_N
ATTACK_S
ATTACK_W
ATTACK_E

HEAL

WAIT
```

但 CandidateBuilder 会过滤掉明显非法动作。

例如：

```text
北边是墙
南边有火
东边有敌人
西边安全
没有血包
```

候选可以变成：

```json
{
  "move_w": "Move one cell west",
  "move_s": "Move one cell south into a fire tile",
  "attack_e": "Attack the adjacent enemy to the east",
  "wait": "Remain in the current cell"
}
```

这里是否保留“危险但合法”的动作很重要：

```text
墙 → 非法 → 不提供
火 → 合法但危险 → 提供
没有敌人的 ATTACK → 不提供
没有血包的 HEAL → 不提供
```

这样 NanoJev 学的是：

> **在合法候选之间做价值判断。**

而不是浪费模型容量学习游戏引擎的基本合法性。

---

# 6. State 设计

第一版**不要截图**。

直接从游戏引擎获取结构化状态。

推荐状态：

```text
tick=183
hp=63/100
score=240

player=(11,8)

nearest_enemy:
direction=E
distance=1
enemy_hp=30

nearest_gem:
direction=NE
distance=4

nearest_medkit:
direction=SW
distance=7

north=safe
south=fire
west=safe
east=enemy

enemies_nearby=1
gems_remaining=4
medkits=0

attack_ready=true
heal_available=false
```

压缩成模型文本：

```text
HP 63/100. Score 240.
Adjacent: N safe, S fire, W safe, E enemy(hp=30).
Nearest gem NE distance 4.
Nearest medkit SW distance 7.
One enemy nearby.
Attack ready. No medkit carried.
```

## Token 预算

目标：

```text
State + Question + Candidates <= 192 tokens
```

最大控制：

```text
<= 256 tokens
```

不要一开始把整个 20×20 地图 ASCII 全塞进去。

NanoJev 应该负责：

```text
局部快速决策
```

而代码负责：

```text
地图规则
记忆
路径规划
合法动作
```

这也和 NanoJev 官方 Maze / Snake 的“模型判断 + 程序规划”思路一致。

---

# 7. NanoJev Request 格式

示意：

```json
{
  "states": [
    {
      "id": "episode_001_tick_183",
      "state": "HP 63/100. Adjacent: N safe, S fire, W safe, E enemy hp30. Nearest gem NE distance 4. Attack ready.",
      "questions": {
        "action": {
          "type": "choice",
          "instructions": "Choose the best immediate action to maximize survival and long-term score.",
          "criteria": {
            "move_n": "Move north to a safe tile",
            "move_s": "Move south onto a fire tile",
            "move_w": "Move west to a safe tile",
            "attack_e": "Attack the adjacent enemy to the east",
            "wait": "Stay in place"
          }
        }
      }
    }
  ]
}
```

返回后只关心：

```text
candidate → probability
```

然后：

```python
action = max(probabilities, key=probabilities.get)
```

第一版直接 greedy。

后面增加：

```text
temperature sampling
epsilon exploration
top-p action sampling
```

---

# 8. 游戏运行机制

## 不要让模型推理卡住 Pygame 60 FPS

把：

```text
Render FPS
```

和：

```text
Decision Tick
```

分开。

推荐 V1：

```text
Render = 60 FPS
Decision = Turn Based
```

即：

```text
Env产生State
↓
等待NanoJev决策
↓
执行1步
↓
刷新动画
↓
下一个State
```

这样最容易：

- replay；
- 评测；
- 数据生成；
- Debug；
- 保证多个 Agent 环境完全一致。

第二阶段再做：

```text
Render = 60FPS
Decision = 2~10Hz
```

并使用后台线程/进程推理。

---

# 9. Reward 设计

第一版建议：

```text
每存活一步             +0.05
吃到 Gem               +10
击杀 Enemy             +20
拾取 Health Pack       +3
受到 1 HP 伤害         -0.1
踩火一次               -3
死亡                    -30
达到最大 Tick 存活      +10
```

最终：

```python
reward =
    survival_reward
    + gem_reward
    + kill_reward
    + heal_reward
    - damage_penalty
    - fire_penalty
    - death_penalty
```

不要第一版搞得太复杂。

Reward 必须：

```text
稳定
可解释
容易计算
无需人工判断
```

因为后面会用它自动生成 NanoJev 的监督分布。

---

# 10. 三个 Baseline Agent

## RandomAgent

从合法动作中随机：

```python
random.choice(actions)
```

用途：

```text
最低基准
```

---

## RuleAgent

类似：

```text
if HP < 25 and HEAL available:
    HEAL

elif adjacent enemy and HP > 35:
    ATTACK

elif immediate danger:
    move to safest tile

elif gem nearby:
    move toward gem

else:
    explore
```

用途：

```text
专家策略
数据生成
Benchmark
```

---

## NanoJevAgent

```text
StateEncoder
↓
CandidateBuilder
↓
NanoJev
↓
Probability Distribution
↓
greedy/sample
↓
Action
```

三个 Agent 必须实现同一个接口：

```python
class Agent:
    def act(self, observation, candidates) -> Action:
        ...
```

---

# 11. 项目目录

建议不要直接把所有代码塞进 NanoJev 仓库。

```text
jev-arena/
│
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ arena.yaml
│  ├─ nanojev_1660s.yaml
│  └─ rewards.yaml
│
├─ arena/
│  ├─ __init__.py
│  ├─ env.py
│  ├─ entities.py
│  ├─ map_generator.py
│  ├─ rewards.py
│  ├─ observation.py
│  ├─ candidates.py
│  └─ renderer.py
│
├─ agents/
│  ├─ base.py
│  ├─ random_agent.py
│  ├─ rule_agent.py
│  └─ nanojev_agent.py
│
├─ nanojev_adapter/
│  ├─ client.py
│  ├─ schema.py
│  ├─ local_predictor.py
│  └─ policy.py
│
├─ datasets/
│  ├─ raw/
│  ├─ generated/
│  └─ splits/
│
├─ scripts/
│  ├─ play.py
│  ├─ benchmark.py
│  ├─ generate_dataset.py
│  ├─ build_soft_targets.py
│  ├─ validate_dataset.py
│  └─ replay.py
│
├─ training/
│  ├─ README.md
│  ├─ patches/
│  └─ configs/
│
├─ runs/
├─ replays/
├─ tests/
│
└─ third_party/
   └─ NanoJev/
```

建议 NanoJev 使用：

```text
Git submodule
```

或者：

```text
单独 clone
```

不要一开始魔改上游目录。

---

# 12. Windows + 1660S 环境

## 推荐

```text
Windows 10/11
Python 3.11
Git
NVIDIA Driver
CUDA 可用的 PyTorch
Pygame
NanoJev
```

不要求：

```text
Docker
WSL
外部 API
OpenAI
TypeSafe Jev API
```

如果 NanoJev 在 Windows 原生遇到 Triton / 编译兼容问题，再切 WSL2。

不是一开始就上 WSL。

---

# 13. 环境验证

PowerShell：

```powershell
nvidia-smi
```

确认：

```text
GeForce GTX 1660 SUPER
```

建立环境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

NanoJev：

```powershell
git clone https://github.com/TianyuCodings/NanoJev.git third_party/NanoJev
```

依赖先按照项目官方版本尝试：

```powershell
pip install -r third_party\NanoJev\requirements-toy.txt
```

再安装：

```powershell
pip install pygame huggingface_hub pyyaml pytest
```

检查 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print(torch.version.cuda)"
```

期望：

```text
True
NVIDIA GeForce GTX 1660 SUPER
<CUDA version>
```

---

# 14. NanoJev 在 1660S 上的第一原则

当前 NanoJev CLI 的 precision 只有：

```text
bf16
fp32
```

所以 **V1 使用 FP32**：

```text
--precision fp32
```

不要：

```text
--precision bf16
```

### 后续优化

等项目跑通后，可以给自己的 fork 增加：

```text
--precision fp16
```

需要修改：

```text
serve_decisions.py
predict_toy_decisions.py
train_pipeline_decisions.py
```

训练 FP16 还应该加：

```text
torch.cuda.amp.GradScaler
```

但这属于：

```text
P2 优化
```

不是 P0 必须项。

---

# 15. 下载 NanoJev 模型

优先下载游戏 checkpoint：

```text
variants/games_gold_seed17
```

因为它已经做过 Snake 类游戏决策。

也保留根 checkpoint 做对照。

模型只需要联网下载一次。

完成后可以启用：

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

然后断网运行。

---

# 16. NanoJev 本地运行策略

第一种：

```text
Jev Arena
↓ HTTP localhost
NanoJev serve_decisions.py
```

优点：

```text
改游戏不会重新加载模型
Agent 和模型解耦
容易 Debug
```

启动：

```powershell
python scripts\serve_decisions.py `
  --checkpoint-dir <checkpoint> `
  --web-root web `
  --host 127.0.0.1 `
  --port 8765 `
  --precision fp32 `
  --disable-native-triton
```

如果不需要：

```text
--disable-native-triton
```

可以去掉。

后面再测试。

---

# 17. Dataset 设计

NanoJev 的训练数据核心：

```text
state
question
criteria
gold_probs
```

Arena 示例：

```json
{
  "id": "arena_seed_100_tick_42",
  "state_id": "arena_seed_100_tick_42",
  "family_id": "arena_v1",
  "split": "train",
  "state": "HP 28/100. E enemy distance1. N safe. S fire. W safe. Nearest gem NE distance3.",
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Choose the best immediate action.",
      "criteria": {
        "move_n": "Move north to a safe tile",
        "move_w": "Move west to a safe tile",
        "attack_e": "Attack the adjacent enemy",
        "move_s": "Move south into fire"
      }
    }
  },
  "gold_probs": {
    "action": {
      "move_n": 0.55,
      "move_w": 0.30,
      "attack_e": 0.10,
      "move_s": 0.05
    }
  },
  "gold_probs_kind": {
    "action": "optimal_action_policy"
  },
  "metadata": {
    "source_group_id": "map_seed_100"
  }
}
```

---

# 18. 不要只做 One-Hot 标签

最简单的是：

```text
RuleAgent 选 ATTACK
↓
ATTACK = 1
其他 = 0
```

可以作为第一批数据。

但更有意思的是：

## Rollout Soft Target

在某个 State：

```text
A = MOVE_N
B = MOVE_W
C = ATTACK_E
D = MOVE_S
```

复制游戏环境：

```text
State
├─执行 A → 模拟未来 8 步 → Return(A)
├─执行 B → 模拟未来 8 步 → Return(B)
├─执行 C → 模拟未来 8 步 → Return(C)
└─执行 D → 模拟未来 8 步 → Return(D)
```

例如：

```text
MOVE_N     17
MOVE_W     13
ATTACK_E    8
MOVE_S     -4
```

通过：

```text
softmax(return / temperature)
```

变成：

```text
MOVE_N     0.56
MOVE_W     0.31
ATTACK_E   0.11
MOVE_S     0.02
```

直接作为：

```text
gold_probs
```

这非常符合 NanoJev 的概率决策训练范式。

而且：

```text
不需要任何大模型 API
不需要人工标注
全部本机程序生成
```

---

# 19. 数据生成规模

V0 Debug：

```text
1,000 states
```

V1：

```text
20,000 states
```

V2：

```text
100,000 states
```

V3：

```text
300,000+
```

不要第一天生成 100 万。

先保证：

```text
数据正确 > 数据很多
```

---

# 20. Train / Dev / Test / OOD

非常重要：

**按地图 seed 分割。**

不要把同一张地图上的不同 tick：

```text
一部分 train
一部分 test
```

那会泄漏。

建议：

```text
Train        70%
Dev          10%
Calibration  5%
Test         10%
OOD           5%
```

OOD 可以专门使用：

```text
更大地图
更多敌人
更多火
低血量开局
不同地图生成器参数
```

metadata：

```json
{
  "source_group_id": "map_seed_1234"
}
```

确保同一个 source_group 不跨 split。

---

# 21. 1660S 训练策略

## Stage A：不训练

先直接测试现有 NanoJev：

```text
Game checkpoint
↓
Arena State
↓
看概率有没有基本语义
```

不要期待它直接会玩。

这一步只是验证：

```text
模型加载
请求成功
概率输出
延迟
显存
```

---

## Stage B：Decision Head Only

这是 1660S 最重要的改造。

NanoJev 当前训练器有：

```text
head warmup
↓
full model updates
```

建议自己的 fork 增加：

```text
--freeze-backbone
```

逻辑：

```python
for p in model.backbone.parameters():
    p.requires_grad_(False)
```

Optimizer：

```python
params = [p for p in model.parameters() if p.requires_grad]
```

最好让 frozen backbone forward：

```python
with torch.no_grad():
    hidden = backbone(...)
```

或者：

```text
显式 detach backbone features
```

这样避免保存整条 Qwen backward activation。

目标：

```text
只训练 NanoJev Decision Head
```

第一批配置：

```text
precision               fp32
max_length              192
microbatch_questions    1
batch_questions         8
max_microbatch_tokens   512~1024
steps                    500
eval_every               50
head_lr                  2e-4
backbone_lr              0
gradient_checkpointing   off
```

如果显存还有余量：

```text
microbatch 1 → 2
```

不要先调大。

---

# 22. Stage C：部分解冻

Head-only 达到瓶颈以后再做。

实验顺序：

```text
Experiment A
Decision Head only

Experiment B
Decision Head + 最后 1 Transformer Block

Experiment C
Decision Head + 最后 2 Transformer Blocks
```

推荐：

```text
max_length = 192
microbatch_questions = 1
gradient_checkpointing = on
```

每次只改变一个变量。

记录：

```text
peak VRAM
step time
train CE
dev CE
test reward
OOD reward
```

**6GB 显存下不要直接全量解冻 0.6B。**

---

# 23. OOM 降级顺序

一旦：

```text
CUDA out of memory
```

按照固定顺序调整：

```text
1. microbatch_questions → 1
2. max_length → 192
3. max_length → 128
4. max_microbatch_tokens ↓
5. 减少候选动作数量
6. Freeze 更多 backbone
7. 开 gradient checkpointing
8. 关闭同时运行的浏览器 / ComfyUI / 其他 GPU 程序
```

不要第一反应去：

```text
缩数据集
```

数据集大小基本不是单步显存的主要问题。

---

# 24. 推理性能优化

第一版：

```text
FP32
batch = 1 state
greedy policy
```

记录：

```text
model_load_seconds
decision_latency_ms
VRAM_peak_MB
GPU_util
```

第二版才考虑：

```text
FP16 inference patch
batch multi-agent
state cache
重复 state hash
```

对于同一个 State：

```text
hash(state + candidates)
↓
LRU Cache
```

如果重复：

```text
直接返回之前概率
```

---

# 25. 多 Agent 并行是 NanoJev 很有意思的点

后面可以一次请求：

```text
state_A
state_B
state_C
state_D
```

NanoJev 做 batch。

Arena 里同时：

```text
🤖 莽夫
🤖 苟王
🤖 贪财
🤖 普通版
```

这样比四个模型逐个请求更高效。

---

# 26. “性格”不要靠 Prompt 糊弄

最好通过 Reward / Dataset 改。

## Berserker

```text
Kill +35
Gem +5
Survive +0.02
Damage -0.03
```

## Survivor

```text
Kill +10
Gem +5
Survive +0.15
Damage -0.3
Death -60
```

## Greedy

```text
Gem +25
Kill +8
Survive +0.03
```

然后：

```text
不同 Reward
↓
不同 rollout gold_probs
↓
不同训练 checkpoint
```

这样形成的策略差异才是真的模型行为差异。

---

# 27. Replay 必须第一版就设计

每一步保存：

```json
{
  "episode": 17,
  "tick": 183,
  "state": {},
  "candidate_actions": [],
  "probabilities": {},
  "chosen_action": "attack_e",
  "reward": 20,
  "hp_before": 63,
  "hp_after": 61,
  "decision_ms": 187,
  "agent": "nanojev",
  "seed": 61005
}
```

这样以后可以：

```text
重放
Debug
画曲线
训练
比较
做 Demo
```

不要只录屏。

---

# 28. Benchmark

固定：

```text
100 个 map seeds
```

每个 Agent 都跑完全相同的 seed。

核心指标：

```text
Mean Reward
Median Reward
Survival Steps
Gems Collected
Kills
Damage Taken
Deaths
Win / Goal Rate
Decision Latency P50
Decision Latency P95
Action Entropy
```

Arena 第一版不一定需要“胜利”。

可以定义：

```text
500 ticks 后仍存活 = episode completed
```

---

# 29. 必须做的对照实验

最终至少跑：

```text
Random
Rule
NanoJev Pretrained
NanoJev Arena Head-only
```

以后增加：

```text
NanoJev Partial-unfreeze
```

你真正需要回答的问题不是：

> NanoJev 能不能动？

而是：

> **训练后的 NanoJev 是否在未见过的地图上学到了可泛化的动作偏好？**

---

# 30. 开发里程碑

## M0 — 环境

完成标准：

```text
torch.cuda.is_available() == True
能识别 GTX 1660 SUPER
NanoJev checkpoint 能加载
本地 health endpoint 返回 ready
```

---

## M1 — Arena 无 AI

完成：

```text
20×20地图
玩家
敌人
宝石
火
血包
墙
碰撞
攻击
HP
Reward
Seed
Replay
```

验收：

```text
RandomAgent 连跑100局不崩
相同seed可完全复现
```

---

## M2 — RuleAgent

完成：

```text
规则决策
安全移动
攻击
回血
追 Gem
```

验收：

```text
Rule > Random
```

如果 Rule 都不能稳定超过 Random：

```text
先修游戏/Reward
不要训练 NanoJev
```

---

## M3 — NanoJev 接入

完成：

```text
StateEncoder
CandidateBuilder
localhost client
概率解析
greedy policy
```

验收：

```text
NanoJev 能连续跑 10 局
非法动作率 = 0
记录每次概率
```

---

## M4 — 数据生成

完成：

```text
Rule label
Rollout soft target
JSONL
split
schema validation
```

验收：

```text
20,000+ states
train/dev/test 无 seed 泄漏
NanoJev validate-only 通过
```

---

## M5 — 1660S Head-only Training

完成：

```text
freeze backbone patch
head-only optimizer
500-step smoke train
保存 checkpoint
```

验收：

```text
无 OOM
loss 正常下降
checkpoint 可重新加载
```

---

## M6 — 正式训练

目标：

```text
100k states
多个 seeds
soft targets
1000~3000 steps
```

验收：

```text
Dev CE 比初始 checkpoint 更低
Arena Reward 比 pretrained 提升
```

---

## M7 — Benchmark Dashboard

输出：

```text
Random
Rule
Pretrained NanoJev
Fine-tuned NanoJev
```

统一 100 个 seed。

---

## M8 — Personality Models

生成：

```text
Berserker
Survivor
Greedy
```

最终做三机器人同屏。

---

# 31. 详细 TODO

下面可以直接作为项目 Issue / GitHub Project 使用。

## P0 — 环境

- [x] 安装 / 更新 NVIDIA Driver
- [x] `nvidia-smi` 正常识别 GTX 1660 SUPER
- [x] 安装 Python 3.12（本机兼容环境）
- [x] 创建 `.venv`
- [x] 安装 CUDA 可用 PyTorch
- [x] `torch.cuda.is_available()` 返回 True
- [x] Clone NanoJev
- [x] 安装 NanoJev requirements
- [x] 下载 `games_gold_seed17` checkpoint
- [x] 使用 `--precision fp32` 启动 NanoJev
- [x] `/api/health` 返回 ready
- [x] 保存一次测试 decision response
- [x] 记录首次模型显存占用
- [x] 记录单次推理 latency

## P0 — Arena Core

- [x] 创建 `ArenaEnv`
- [x] 支持固定 random seed
- [x] 创建 Grid Map
- [x] 实现 Wall
- [x] 实现 Player
- [x] 实现 Enemy
- [x] 实现 Gem
- [x] 实现 Fire
- [x] 实现 Health Pack
- [x] 实现 HP
- [x] 实现攻击
- [x] 实现受伤
- [x] 实现死亡
- [x] 实现 Gem 得分
- [x] 实现 max_ticks
- [x] 实现 Reward
- [x] 实现 `reset(seed)`
- [x] 实现 `step(action)`
- [x] 实现 `clone()`
- [x] 实现 terminal
- [x] 写基本单元测试

## P0 — Pygame

- [x] 创建 60 FPS Renderer
- [x] 地图渲染
- [x] Player 渲染
- [x] Enemy 渲染
- [x] Gem 渲染
- [x] Fire 渲染
- [x] Health 渲染
- [x] 显示 HP
- [x] 显示 Score
- [x] 显示 Tick
- [x] 显示 Agent 类型
- [x] 支持 Pause
- [x] 支持 Restart
- [x] 支持 Seed 输入
- [x] 支持加速回放
- [x] 收齐宝石自动进入下一关
- [x] 随关卡提高敌人、火焰、墙和伤害并减少补给

## P0 — Agent Interface

- [x] 定义 `Agent.act()`
- [x] RandomAgent
- [x] RuleAgent
- [x] NanoJevAgent skeleton
- [x] 所有 Agent 使用同一 Observation
- [x] 所有 Agent 使用同一 Candidate Set

## P0 — Observation

- [x] 玩家 HP
- [x] Adjacent tiles
- [x] 最近 Enemy
- [x] 最近 Gem
- [x] 最近 Health Pack
- [x] 当前危险
- [x] attack_ready
- [x] heal_available
- [x] 生成 compact text
- [x] 统计 token 长度
- [x] 保证 P95 state <= 192 tokens

## P0 — CandidateBuilder

- [x] 墙方向不提供 MOVE
- [x] 地图外不提供 MOVE
- [x] 只有 adjacent enemy 才提供 ATTACK
- [x] 有回血条件才提供 HEAL
- [x] WAIT 永远存在
- [x] Fire movement 保持“合法但危险”
- [x] Candidate ID 保持稳定
- [x] Candidate 描述包含动作结果
- [x] 单元测试所有边界情况

## P0 — RuleAgent

- [x] HP 危险时优先 HEAL
- [x] adjacent enemy 攻击逻辑
- [x] 火焰规避
- [x] Gem 导向
- [x] 简单 exploration
- [x] 防止两格循环
- [x] 固定 seed 可复现
- [x] 100 episodes benchmark
- [x] 验证 Rule > Random

## P0 — Replay

- [x] Episode JSONL
- [x] 保存 state
- [x] 保存 candidates
- [x] 保存 probabilities
- [x] 保存 chosen_action
- [x] 保存 reward
- [x] 保存 latency
- [x] 保存 seed
- [x] 保存 HP
- [x] 保存 score
- [x] 支持 deterministic replay

## P1 — NanoJev Integration

- [x] 实现 localhost client
- [x] 构造 NanoJev request schema
- [x] Parse Choice probability
- [x] greedy policy
- [x] timeout
- [x] retry 1 次
- [x] 模型失败时暂停而不是偷偷切 RuleAgent
- [x] UI 展示动作概率
- [x] UI 展示推理耗时
- [x] 连续运行10局
- [x] 连续运行1000 decisions 不崩

## P1 — 数据生成

- [x] Episode sampler
- [x] 随机地图 generator
- [x] RuleAgent teacher
- [x] 收集 state/candidates/action
- [x] one-hot gold_probs
- [x] rollout evaluator
- [x] state.clone()
- [x] 对每个 candidate 做 N-step rollout
- [x] return → softmax
- [x] 生成 soft `gold_probs`
- [x] `gold_probs_kind=optimal_action_policy`
- [x] source_group_id=map_seed
- [x] Train split
- [x] Dev split
- [x] Calibration split
- [x] Test split
- [x] OOD split
- [x] Dataset manifest
- [x] Validate probabilities sum=1
- [x] Validate candidate keys 完全一致
- [x] Validate 无 seed leakage
- [x] 首批 1k dataset
- [x] 首批 20k dataset
- [x] 正式 100k dataset

## P1 — 1660S Training Patch

- [ ] Fork NanoJev
- [x] 增加 `--freeze-backbone`
- [x] Backbone `requires_grad=False`
- [x] Optimizer 只接受 trainable params
- [x] Frozen backbone 不保存 backward graph
- [x] 输出 trainable parameter count
- [x] 输出 peak VRAM
- [x] 增加 1660S config
- [x] `max_length=192`
- [x] `microbatch_questions=1`
- [x] `batch_questions=8`
- [x] `precision=fp32`
- [x] 50-step smoke test
- [x] 500-step smoke test
- [x] checkpoint reload test
- [x] evaluation test

## P1 — Benchmark

- [x] 固定 100 map seeds
- [x] Random benchmark
- [x] Rule benchmark
- [ ] Pretrained NanoJev benchmark
- [ ] Fine-tuned NanoJev benchmark
- [x] Mean Reward
- [x] Median Reward
- [x] Survival
- [x] Gems
- [x] Kills
- [x] Damage
- [x] Death
- [x] P50 latency
- [x] P95 latency
- [x] Action entropy
- [x] 导出 CSV/JSON
- [x] 生成对照图

## P2 — 性能

- [ ] 研究 FP16 inference
- [ ] NanoJev CLI 添加 `fp16`
- [ ] 验证概率结果数值稳定
- [ ] benchmark FP32 vs FP16
- [x] LRU State Cache
- [x] batch multi-agent states
- [x] 后台 inference worker
- [x] Render / Decision 解耦
- [ ] Decision 2Hz
- [ ] Decision 5Hz
- [ ] Decision 10Hz

## P2 — Partial Fine-tune

- [ ] Head + last block
- [ ] Gradient checkpointing
- [ ] microbatch=1
- [ ] 记录 peak VRAM
- [ ] Head + last 2 blocks
- [ ] 与 head-only 对照
- [ ] 如果频繁 OOM，停止该路线
- [ ] 再考虑 LoRA/PEFT patch

## P2 — Personality

- [ ] Berserker reward
- [ ] Survivor reward
- [ ] Greedy reward
- [ ] 生成三份 soft target
- [ ] 训练三个 checkpoint
- [ ] 同一 100 seed benchmark
- [ ] 三 Agent 同屏
- [ ] 概率条对照
- [ ] 自动生成比赛 replay

---

# 32. 第一周建议顺序

不要按“AI 最酷的部分”开始。

正确顺序：

```text
Day 1
CUDA + NanoJev 本地 checkpoint 跑起来

Day 2
ArenaEnv + RandomAgent

Day 3
RuleAgent + Reward + Replay

Day 4
NanoJevAgent 接入

Day 5
Dataset Generator

Day 6
Head-only Patch + 1k smoke dataset

Day 7
20k Dataset + 第一个 Fine-tuned checkpoint
```

关键：

> **第 1~4 天完全不用训练。**

先证明：

```text
环境正确
状态正确
候选正确
Reward 正确
NanoJev 能稳定调用
```

再训练。

---

# 33. Definition of Done：V1

V1 完成必须同时满足：

```text
[ ] 完全不调用外部模型 API
[ ] 断网以后仍可运行
[ ] GTX 1660S 可以推理
[ ] Pygame 游戏稳定运行
[ ] Random / Rule / NanoJev 三 Agent 可切换
[ ] NanoJev 返回真实概率分布
[ ] 概率可视化
[ ] Replay 完整保存
[ ] 可生成训练 JSONL
[ ] 1660S 可进行 head-only 训练
[ ] Fine-tuned checkpoint 可重新加载
[ ] 固定100地图完成统一 Benchmark
```

只要这些完成，这就已经是一个完整 GitHub 项目，而不是 Demo 脚本。

---

# 34. V1 暂时明确不做

以下全部放后面：

```text
× 摄像头
× 游戏截图视觉识别
× YOLO
× OCR
× Doom
× 云顶之弈
× MCTS
× LLM System-2
× LangGraph
× 多机
× RLHF
× 在线 API
× 全参数微调
```

原因：

这些会把一个可以完成的项目变成：

```text
十几个子项目同时施工
```

先把：

```text
Structured State → NanoJev → Action
```

做到非常扎实。

---

# 35. V2 以后怎么进化

V1：

```text
Structured State
↓
NanoJev
↓
Action
```

V2：

```text
Structured State
↓
NanoJev
↓
多 Agent 对战
```

V3：

```text
State
↓
MCTS
↓
候选未来状态
↓
NanoJev 评估
```

V4：

```text
LLM System-2
    ↓
战略目标
    ↓
NanoJev System-1
    ↓
实时动作
```

V5：

```text
Screenshot
↓
Vision Encoder
↓
Structured State / Embedding
↓
NanoJev
```

到 V5 再考虑：

```text
真实游戏控制
Doom
麻将
云顶之弈
```

---

# 36. 最重要的工程原则

整个项目只坚持四件事：

```text
1. Game Engine 决定事实。
2. CandidateBuilder 决定什么动作合法。
3. NanoJev 决定合法动作里更想选哪个。
4. Replay Logger 记录一切。
```

不要让 NanoJev：

```text
计算碰撞
记地图
解析坐标规则
控制 UI
维护 HP
判断游戏结束
```

这些确定性的东西全部交给代码。

NanoJev 专心做：

> **在一个复杂但有限的动作集合中，快速进行概率决策。**

这才是这个项目真正要验证的东西。

---

# 37. 推荐第一条开发分支

```text
main
└─ feat/arena-core
   └─ feat/rule-agent
      └─ feat/nanojev-local
         └─ feat/dataset-generator
            └─ feat/1660s-head-training
```

第一个可发布版本：

```text
v0.1.0
Jev Arena — Local NanoJev on GTX 1660 Super
```

Release 至少附：

```text
README
GIF / MP4
100-seed Benchmark
Hardware
VRAM
Latency
Training config
Dataset size
Checkpoint
```

---

# 38. 你真正应该先完成的 10 个任务

如果只看这一段，就按这个顺序做：

```text
01. CUDA / PyTorch 识别 1660S
02. NanoJev FP32 checkpoint 本地推理成功
03. ArenaEnv 能跑
04. RandomAgent 连跑100局
05. RuleAgent 明显优于 Random
06. Replay 保存完整轨迹
07. NanoJevAgent 接入并展示概率条
08. 生成 20k JSONL 训练数据
09. 给 NanoJev 加 freeze-backbone，1660S head-only 训练
10. Random / Rule / NanoJev / Fine-tuned NanoJev 跑统一100-seed Benchmark
```

做完第 10 项，这个项目就已经有实际研究价值和展示价值。

---

## Sources / 基础依据

NanoJev README：

https://github.com/TianyuCodings/NanoJev/blob/main/README.zh-CN.md

NanoJev Pipeline：

https://github.com/TianyuCodings/NanoJev/blob/main/research/pipeline_runbook.md

NanoJev requirements：

https://github.com/TianyuCodings/NanoJev/blob/main/requirements-toy.txt

NVIDIA GTX 16 系列规格：

https://www.nvidia.com/en-gb/geforce/graphics-cards/compare/

NVIDIA CUDA Compute Capability：

https://developer.nvidia.com/cuda/gpus
