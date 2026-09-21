# Jev Arena V2 战术博弈升级规划

> 项目：`liao96312/jev-arena-nanojev`  
> 目标：把当前“寻路 + 躲怪 + 拿宝石”的 Arena，升级为一个适合 NanoJev / Rule / Search Agent 做战术推理与长期规划对比的本地实验场。  
> 核心原则：**不要通过单纯增加怪物数量制造难度，而要通过敌人意图、位移、环境互动、技能冷却和多步规划制造博弈。**

---

## 1. 当前版本问题诊断

当前 Arena 的主要决策动作是：

```text
move_n / move_s / move_w / move_e
attack_n / attack_s / attack_w / attack_e
heal
wait
```

实际每个状态中，通常只有 3~5 个有效候选动作。

现有战斗逻辑的主要问题：

1. 敌人同质化，仅有位置和 HP。
2. 敌人行为基本为追击玩家。
3. 玩家无法主动改变敌人位置。
4. 火焰等地形主要是玩家要规避的障碍，而不是可利用的战斗资源。
5. 玩家缺少 Dash、控制、击退等“改变局面”的技能。
6. 怪物增加时，安全格数量减少，容易从“更难”直接变成“无解”。
7. `hybrid` 策略大量依赖 BFS 宝石路线，NanoJev 更多是在同类路径之间做选择。
8. 数据生成使用 `RuleAgent` 作为 rollout teacher，训练数据上限受规则代理能力限制。
9. 当前 Observation 更偏“最近目标摘要”，不足以表达敌人未来行为和战术关系。

当前难度增长方式更接近：

```text
怪物数量增加
    ↓
可走空间减少
    ↓
安全候选减少
    ↓
死局增加
```

V2 目标应改成：

```text
怪物数量增加
    ↓
敌人意图组合增加
    ↓
敌我互动关系增加
    ↓
玩家可利用的连锁关系增加
    ↓
决策树扩大
```

---

# 2. V2 总体设计目标

V2 的核心循环：

```text
观察敌人意图
    ↓
预测未来 1~3 回合威胁
    ↓
移动 / 推动 / 攻击 / Dash / 控制
    ↓
利用地形和敌人互相伤害
    ↓
敌人执行 Intent
    ↓
状态变化
    ↓
重新规划
```

最终希望同一个局面可以同时存在多种合理策略：

```text
方案 A：直接击杀高威胁敌人
方案 B：推怪进火焰
方案 C：诱导冲锋怪撞另一个敌人
方案 D：EMP 控场后抢宝石
方案 E：承受一次伤害换取高价值目标
方案 F：Dash 脱离包围，下一轮再反击
```

---

# 3. V2.0：敌人 Intent 系统

## 3.1 目标

让敌人不再“玩家走一步，它立即追一步”，而是提前暴露未来行为。

这是整个 V2 最重要的基础能力。

## 3.2 数据结构

修改：

```text
arena/entities.py
```

建议：

```python
from dataclasses import dataclass
from enum import StrEnum


class EnemyType(StrEnum):
    CHASER = "chaser"
    CHARGER = "charger"
    ARCHER = "archer"
    BOMBER = "bomber"


class IntentType(StrEnum):
    MOVE = "move"
    MELEE = "melee"
    CHARGE = "charge"
    SHOOT = "shoot"
    EXPLODE = "explode"
    WAIT = "wait"


@dataclass
class Intent:
    kind: IntentType
    direction: str | None = None
    countdown: int = 1
    power: int = 0


@dataclass
class Enemy:
    position: tuple[int, int]
    hp: int = 30
    enemy_type: EnemyType = EnemyType.CHASER
    intent: Intent | None = None
    stunned: int = 0
```

## 3.3 回合流程

当前：

```text
Player Action
Enemy Move
```

V2.0：

```text
Round Start
↓
生成 / 保留 Enemy Intent
↓
Player Action
↓
Intent countdown -= 1
↓
countdown == 0 的敌人执行行为
↓
重新生成 Intent
```

第一版建议所有敌人 Intent 完全可见。

## 3.4 Observation 增加

修改：

```text
arena/observation.py
```

新增类似：

```text
Enemy 1: charger at NE distance 4, hp=30, intent=charge_w, countdown=1
Enemy 2: archer at N distance 5, hp=20, intent=shoot_s, countdown=2
Enemy 3: bomber at SW distance 3, hp=25, intent=explode, countdown=2
```

第一阶段不要只给“最近敌人”。

建议 Observation 至少保留最近 6 个敌人的完整战术摘要。

## 3.5 TODO

- [x] 新增 `EnemyType`
- [x] 新增 `IntentType`
- [x] 新增 `Intent`
- [x] Enemy 保存 `enemy_type / intent / stunned`
- [x] `reset()` 时按比例生成不同敌人
- [x] 新增 `_plan_enemy_intents()`
- [x] 新增 `_resolve_enemy_intents()`
- [x] 移除旧 `_move_enemies()` 的直接追击主逻辑
- [x] Observation 输出 Intent
- [x] Replay 保存 Intent
- [x] Renderer 显示 Intent 图标 / 箭头 / countdown
- [x] 添加 deterministic seed 测试

## 3.6 验收标准

- 相同 seed 下，敌人类型与 Intent 完全可复现。
- GUI 能提前看到敌人下一次攻击方向。
- 至少 90% 的战斗状态存在不止一个合法应对动作。
- 玩家能依据 Intent 主动规避，而不是只能被动逃跑。

---

# 4. V2.1：Shove 位移系统

## 4.1 目标

让玩家能够主动改变敌人位置。

新增动作：

```text
shove_n
shove_s
shove_w
shove_e
```

## 4.2 规则

基础规则：

```text
玩家相邻格有敌人
↓
目标敌人后方一格为空
↓
可以 Shove
↓
敌人被推动 1 格
```

若后方为：

```text
Wall      → 撞墙伤害
Fire      → 进入火焰并受到伤害
Enemy     → 发生碰撞，可选造成双方伤害
Pit       → 后续版本可直接击杀
Barrel    → 后续版本触发爆炸
```

## 4.3 建议新增 Action

```python
SHOVE_N = "shove_n"
SHOVE_S = "shove_s"
SHOVE_W = "shove_w"
SHOVE_E = "shove_e"
```

## 4.4 TODO

- [x] Action 新增四方向 Shove
- [x] `legal_actions()` 动态判断能否推
- [x] 新增 `_push_entity()`
- [x] 支持推怪入火
- [x] 支持推怪撞墙
- [x] 支持推怪撞怪
- [x] candidate description 描述推动结果
- [x] Renderer 增加 Shove 动画或事件文字
- [x] Dataset 支持 Shove candidate

## 4.5 候选描述示例

```text
Shove east enemy one cell east into fire; expected 10 fire damage
```

或者：

```text
Shove north enemy into another enemy; both may take collision damage
```

## 4.6 验收标准

至少出现以下局面：

```text
🔥 E P
```

玩家可通过：

```text
shove_w
```

让敌人进入火焰。

---

# 5. V2.2：环境伤害统一化

## 5.1 目标

环境不再只伤害玩家。

将：

```text
player enters fire → damage player
```

改成：

```text
any damageable entity enters fire → damage entity
```

## 5.2 统一伤害接口

建议新增：

```python
def damage_entity(entity, amount, source):
    ...
```

或者抽象：

```python
class Damageable:
    hp: int
```

第一版不必做复杂继承，只需要统一 Player / Enemy 的伤害处理即可。

## 5.3 新增环境

第一阶段：

```text
Fire
Wall collision
```

第二阶段：

```text
Spike
Pit
Explosive Barrel
Poison
Electric Tile
```

## 5.4 TODO

- [x] `_damage()` 支持 Player / Enemy
- [x] Fire 能伤害 Enemy
- [x] Enemy 可因环境伤害死亡
- [x] 环境击杀计入 kills / score
- [x] Replay 记录 damage source
- [x] Dataset reward 区分直接击杀和环境击杀

---

# 6. V2.3：敌人异质化

第一版仅做 4 种敌人，不要一次做太多。

---

## 6.1 Chaser

基础近战敌人。

行为：

```text
向玩家靠近
相邻后准备 melee
```

作用：

```text
基础空间压力
```

---

## 6.2 Charger

核心战术怪。

行为：

```text
提前显示冲锋方向
1 回合后沿直线冲锋 2~4 格
```

可能结果：

```text
撞玩家
撞墙
撞怪
冲进火焰
冲过头
```

关键要求：

**Charger 必须能够撞其它怪。**

这样玩家才能诱导敌人互殴。

---

## 6.3 Archer

远程压力怪。

行为：

```text
锁定一条直线
1~2 回合后射击
```

射线可被：

```text
墙
敌人
玩家
```

阻挡。

关键要求：

**射手必须可以射到其它敌人。**

否则其战术价值仍然很低。

---

## 6.4 Bomber

高价值连锁敌人。

行为：

```text
进入蓄爆状态
countdown=2
↓
范围 1~2 格爆炸
```

爆炸影响：

```text
玩家
敌人
爆炸桶
部分地图机关
```

Bomber 是解决“怪越多越死局”的重要设计。

怪物多时，Bomber 可能变成玩家最强的临时武器。

## 6.5 TODO

- [x] Chaser Intent
- [x] Charger Intent
- [x] Archer Intent
- [x] Bomber Intent
- [ ] 每种敌人独立 HP / damage / cooldown
- [x] 地图生成保持敌人组成合理
- [x] Renderer 不同图标
- [x] Observation 标明 enemy_type
- [x] enemy-friendly-fire 测试

---

# 7. V2.4：玩家技能系统

第一版只做三个技能。

不要加入十几种技能。

---

## 7.1 Dash

```text
dash_n
ndash_s
ndash_w
ndash_e
```

> 实际代码名称建议统一为 `dash_n / dash_s / dash_w / dash_e`。

效果：

```text
沿方向移动 2 格
中间不可穿墙
可用于脱离包围
```

建议：

```text
CD = 3
```

---

## 7.2 Shove

V2.1 已实现。

建议：

```text
CD = 0 或 1
```

前期最好 CD=0，方便测试位移机制。

---

## 7.3 EMP

效果：

```text
周围 Manhattan Distance <= 1 的敌人
stunned += 1
```

建议：

```text
CD = 4
```

用途：

```text
解围
阻止 Bomber 爆炸
延后 Charger
制造抢宝石窗口
```

---

## 7.4 Player 数据结构

```python
@dataclass
class Player:
    position: tuple[int, int]
    hp: int = 100
    medkits: int = 0
    cooldowns: dict[str, int] = field(default_factory=dict)
```

## 7.5 动态候选

禁止把所有技能永远塞进 candidates。

应按状态动态出现。

典型状态候选数量控制在：

```text
6 ~ 10
```

最大尽量：

```text
<= 12
```

避免 NanoJev 输入暴涨。

## 7.6 TODO

- [x] Player cooldown 数据结构
- [x] 每回合 cooldown tick
- [x] Dash 合法性判断
- [x] EMP 范围计算
- [x] Candidate 描述剩余 CD
- [x] Observation 输出 CD
- [x] Renderer 显示技能状态
- [x] Dataset 支持技能动作

---

# 8. V2.5：2 AP 回合系统

## 8.1 目标

让单回合出现技能组合。

当前：

```text
Player Action
Enemy Action
```

改为：

```text
Round Start
AP = 2
↓
Player Action 1
↓
Player Action 2
↓
Enemy Intent Resolve
```

## 8.2 为什么需要 AP

否则：

```text
Shove
↓
敌人行动
```

玩家很难形成组合。

2 AP 后可以：

```text
Shove + Move
EMP + Attack
Dash + Attack
Attack + Retreat
Shove + Shove
```

## 8.3 重要原则

不要把组合动作编码成：

```text
shove_e_then_dash_n
```

否则动作空间会组合爆炸。

正确做法：

```text
Agent 每次只选一个 action
Env 保存 remaining_ap
remaining_ap == 0 后才 resolve enemy
```

## 8.4 State 新字段

```text
round
ap_remaining
```

## 8.5 TODO

- [x] Arena 增加 round
- [x] Arena 增加 ap_remaining
- [x] 每 Round 重置 AP
- [x] AP=0 后 resolve enemy intents
- [x] WAIT 消耗 1 AP
- [x] HEAL 消耗 1 AP
- [x] 技能分别配置 AP cost
- [x] Observation 输出 AP
- [x] Replay 保存 round/AP
- [x] GUI 显示 AP

---

# 9. V2.6：互动地形

等基础战术机制稳定后再加。

优先级建议：

## 9.1 Explosive Barrel

```text
HP = 1
受伤后爆炸
范围伤害
可连锁其它 Barrel
```

这是最值得加的环境元素。

可形成：

```text
Shove Enemy → Barrel
↓
Barrel Explodes
↓
Bomber damaged
↓
Bomber explodes
↓
Area clear
```

## 9.2 Spike

```text
进入即受到伤害
```

## 9.3 Pit

```text
普通移动不可进入
被 Shove / Charge 推入 → instant kill
```

Pit 会显著提高位移技能价值。

## 9.4 Door / Switch

后期加入。

例如：

```text
踩开关
↓
开启一条路线
关闭另一条路线
```

适合测试长期规划。

---

# 10. 地图生成器 V2

当前随机撒点容易制造天然无解地图。

V2 需要从“随机生成”升级成“随机生成 + 可玩性约束”。

## 10.1 必须检测

生成地图后检查：

```text
玩家至少有 2 个基础可移动方向
至少 1 个 gem reachable
不存在玩家出生即被必杀
至少存在一个安全状态序列
```

## 10.2 可选 Solver 检测

用 BFS / Beam Search 验证：

```text
未来 N=4~8 步内
是否存在 HP > 0 的轨迹
```

如果没有：

```text
重新生成地图
```

## 10.3 不要保证地图“很容易”

只保证：

```text
有解
```

而不是：

```text
简单
```

---

# 11. Observation V2

现有 Observation 太像摘要。

V2 应加入：

```text
HP
AP
cooldowns
gems remaining
medkits
nearest tactical terrain
all relevant enemy intents
hazard lines
nearby interactive objects
```

示例：

```text
HP 45/100. AP 2/2.
Cooldowns: dash=1, emp=0.
Player at (8,9).

Enemies:
1. charger at E distance 3, hp=30, intent=charge_w, countdown=1
2. archer at N distance 4, hp=20, intent=shoot_s, countdown=2
3. bomber at SW distance 2, hp=15, intent=explode radius1, countdown=1

Environment:
Fire west distance 1.
Barrel east distance 2.
Gem north-east distance 4.

Immediate threats:
charger will cross player row next resolve
bomber explosion threatens current tile
```

最后一段 `Immediate threats` 可以先由规则代码生成，而不是 LLM 自己推导。

这样可以控制 token。

已实现：状态输出玩家位置、Dash CD、四邻格、最近战术目标、最近 3 只敌人的
类型/意图/倒计时，以及由环境规则统一计算的全部即时威胁。100 条 rollout 的
候选路径 token 审计为 P50/P95/最大值 `174/184/191`，兼容 `max_length=192`。

---

# 12. Candidate Description V2

Candidate 不要只写：

```text
Move one cell east
```

需要描述立即可计算后果。

例如：

```text
move_n:
Move north to safe tile; leaves bomber blast radius; nearest gem distance 4→3.
```

```text
shove_w:
Push adjacent chaser west into fire; expected 10 fire damage; chaser hp 20→10.
```

```text
emp:
Stun 2 nearby enemies for 1 round; prevents bomber explosion this resolve; cooldown becomes 4.
```

```text
dash_s:
Dash south two cells; escapes charger line; gem distance 3→5; cooldown becomes 3.
```

原则：

**只提供立即可以确定的信息，不要在 candidate 里写复杂策略判断。**

否则规则系统会替 NanoJev 做完决策。

已实现：每个合法动作会在克隆环境中执行一回合，只输出实际发生的 HP、宝石、
击杀、环境击杀、终局变化和下一次 Intent 伤害；Dash 候选同时输出剩余 CD。

---

# 13. Reward V2

当前 reward 偏向：

```text
拿 gem
击杀
活着
```

V2 要重新整理。

建议初版：

```text
Gem                 +10
Kill                 +8
Environment Kill    +10
Level Complete      +25
Damage Taken        -0.15 * damage
Death               -40
Waste Skill          -0.5
Friendly Hazard Use +1~2 shaping
```

注意：

不要对：

```text
Shove
EMP
Dash
```

本身直接给大量奖励。

奖励应该看结果，而不是技能名称。

否则模型会学成“技能能放就放”。

---

# 14. 数据生成 V2

当前：

```text
Candidate
↓
第一步尝试
↓
RuleAgent rollout 8 step
↓
return
```

V2 初期先保持兼容，但改进 RuleAgent。

## 14.1 RuleAgent V2

至少理解：

```text
即将受到的伤害
敌人 Intent
Shove 环境击杀
Bomber AoE
Dash 脱困
EMP 解围
```

否则 rollout teacher 会严重误导训练。

## 14.2 Rollout Horizon

建议测试：

```text
4
8
12
```

不要直接盲目加到 30。

因为环境分支增多后计算量会显著提高。

---

# 15. V3：MCTS / Beam Search Teacher

V2 环境稳定后，再做真正的 Search Teacher。

这是下一阶段最有研究价值的部分。

---

## 15.1 为什么换掉 RuleAgent rollout

当前 teacher 上限：

```text
RuleAgent 能想到什么
NanoJev 才有机会学到什么
```

希望升级为：

```text
Search 能搜索出的策略
↓
变成训练分布
↓
NanoJev 学习搜索结果
```

---

## 15.2 Beam Search 第一版

在 MCTS 之前，可以先做 Beam Search。

优点：

```text
实现简单
Deterministic Arena 很适合
调试容易
容易保存轨迹
```

参数建议：

```text
Depth = 6
Beam Width = 16
```

状态评分：

```text
score =
    hp_weight
  + gem_weight
  + kill_weight
  - threat_weight
  + mobility_weight
  + future_damage_weight
```

先作为 Teacher，不用于 Runtime。

---

## 15.3 MCTS 第二版

接口建议：

```python
search(env) -> {
    "action_probs": {...},
    "action_values": {...},
    "visits": {...}
}
```

Dataset 保存：

```json
{
  "gold_probs": {
    "action": {
      "move_n": 0.12,
      "shove_e": 0.54,
      "emp": 0.24,
      "wait": 0.10
    }
  }
}
```

而不是 one-hot。

---

# 16. Benchmark V2

只看平均 reward 已经不够。

新增指标：

```text
Average Reward
Median Reward
Win Rate
Death Rate
Gem Completion Rate
Average HP Remaining
Average Damage Taken
Kills
Environment Kills
Skill Efficiency
Deadlock Rate
Unique Action Rate
Decision Entropy
Average Branch Factor
P95 Decision Latency
```

最关键的是：

```text
Average Branch Factor
Deadlock Rate
```

因为 V2 的目标就是扩大真实可决策空间。

---

# 17. 博弈复杂度指标

建议增加一个专门的脚本：

```text
scripts/analyze_game_complexity.py
```

统计：

```text
平均合法动作数
P50/P95 合法动作数
仅 1 个可行动作状态比例
2 个动作状态比例
>= 6 个动作状态比例
立即必死状态比例
未来 2 回合必死状态比例
动作价值差距
策略熵
```

目标：

当前可能类似：

```text
Avg Branch Factor: 3.7
Deadlock Rate: 18%
```

V2 目标：

```text
Avg Branch Factor: 6~9
Deadlock Rate: < 5%
```

实际目标以 benchmark 为准。

---

# 18. Agent 对比矩阵

最终建议支持：

```text
RandomAgent
RuleAgentV2
NanoJevModelOnly
NanoJevHybrid
BeamSearchAgent
MCTSAgent
NanoJev + Search
```

核心研究问题：

```text
NanoJev 是否学到了 Search Teacher 的策略分布？

NanoJev 是否能以 1/10 或更低计算成本接近 Search Agent？

复杂度提升后 NanoJev 是否仍优于 RuleAgent？
```

---

# 19. 推荐目录结构

```text
arena/
    env.py
    entities.py
    actions.py
    intents.py
    combat.py
    hazards.py
    mapgen.py
    candidates.py
    observation.py
    renderer.py

agents/
    random_agent.py
    rule_agent.py
    beam_agent.py
    mcts_agent.py
    nanojev_agent.py

search/
    evaluator.py
    beam.py
    mcts.py

scripts/
    play.py
    play_gui.py
    benchmark.py
    analyze_game_complexity.py
    generate_dataset.py

configs/
    arena_v2_easy.py
    arena_v2_normal.py
    arena_v2_hard.py
```

不要求一次重构完成。

优先保证功能，再拆文件。

---

# 20. 具体开发阶段

## Phase 0：冻结 V1

预计目标：保存当前版本作为 baseline。

TODO：

- [x] 打 Git tag：`v1-baseline`
- [x] 保存当前 100 seed benchmark
- [ ] 保存 Random / Rule / NanoJev baseline
- [x] 保存 action branch factor 统计
- [x] 保留当前 dataset manifest

验收：

```text
以后所有 V2 改动均能与 V1 对照。
```

---

## Phase 1：Intent

TODO：

- [x] EnemyType
- [x] Intent
- [x] countdown
- [x] GUI telegraph
- [x] Observation
- [x] tests

验收：

```text
玩家可以看到下一次敌人行为。
```

---

## Phase 2：Shove + 环境伤害

TODO：

- [x] shove action
- [x] push resolver
- [x] enemy fire damage
- [x] wall collision
- [x] environmental kill
- [x] tests

验收：

```text
玩家能利用地图杀敌。
```

---

## Phase 3：四类敌人

TODO：

- [x] Chaser
- [x] Charger
- [x] Archer
- [x] Bomber
- [x] friendly fire
- [x] intent renderer
- [x] tests

验收：

```text
不同敌人的处理优先级明显不同。
```

---

## Phase 4：技能

TODO：

- [x] Dash
- [x] EMP
- [x] Cooldown
- [x] Dynamic Candidates
- [x] skill UI

验收：

```text
包围局面不再天然等于死亡。
```

---

## Phase 5：2 AP

TODO：

- [x] round
- [x] AP
- [x] resolve timing
- [x] GUI
- [x] Replay

验收：

```text
出现 Shove+Move、EMP+Attack 等组合。
```

---

## Phase 6：地图 V2

TODO：

- [x] Barrel
- [ ] Spike
- [ ] Pit
- [x] map solver
- [x] dead-map regeneration

验收：

```text
随机地图不会频繁生成天然必死局。
```

---

## Phase 7：RuleAgent V2 + Dataset

TODO：

- [x] Threat scoring
- [x] Intent awareness
- [x] Environmental kill awareness
- [x] Skill usage
- [x] rollout tests
- [x] generate 10k smoke dataset

验收：

```text
RuleAgent V2 明显高于 Random。
```

---

## Phase 8：NanoJev V2

TODO：

- [x] audit tokens
- [x] 调整 max_length（压缩观测后沿用 192）
- [x] 10k dataset smoke train
- [ ] 100k dataset train
- [ ] benchmark
- [ ] calibration

验收：

```text
NanoJev 不依赖 BFS planner 也能处理部分战术局面。
```

---

## Phase 9：Beam Teacher

TODO：

- [ ] state evaluator
- [ ] depth search
- [ ] transposition cache
- [ ] action values
- [ ] gold distribution

验收：

```text
Beam > RuleAgentV2
```

---

## Phase 10：MCTS

TODO：

- [ ] node
- [ ] selection
- [ ] expansion
- [ ] rollout/eval
- [ ] backprop
- [ ] visit policy
- [ ] dataset export

验收：

```text
Search teacher 可稳定输出 action distribution。
```

---

# 21. 第一轮不要做的东西

以下内容先不要做：

```text
20+ 技能
几十种怪物
复杂装备系统
随机词条
角色成长
联网
多人对战
3D
大型地图
复杂剧情
LLM 生成地图
```

这些都会稀释项目核心。

核心应该始终是：

```text
小模型决策能力实验
```

---

# 22. 推荐最小 V2 MVP

如果希望尽快看到质变，只做下面 6 件事：

```text
1. Enemy Intent
2. Charger
3. Bomber
4. Shove
5. Enemy 可受 Fire / Friendly Fire
6. Dash
```

甚至先不要做 Archer 和 EMP。

完成后就能出现：

```text
诱导 Charger
↓
撞 Bomber
↓
Bomber 爆炸
↓
炸掉其它敌人
↓
玩家 Dash 离场
```

这种局面一出现，Arena 就已经从寻路游戏变成战术游戏了。

---

# 23. MVP 推荐开发顺序

```text
Day 1
EnemyType + Intent

Day 2
Charger

Day 3
Shove

Day 4
Enemy fire damage + collision

Day 5
Bomber

Day 6
Dash

Day 7
Observation + Candidates

Day 8
Renderer

Day 9
Tests

Day 10
RuleAgentV2

Day 11
Benchmark

Day 12
Dataset smoke generation

Day 13+
NanoJev retrain
```

这里的 Day 只是开发顺序，不是硬性工期。

---

# 24. 推荐首批测试场景

不要只随机测。

新增固定 tactical scenarios。

目录建议：

```text
tests/scenarios/
```

## Scenario 1：Push into Fire

```text
#####
#...#
#FEP#
#...#
#####
```

最优动作应包含：

```text
shove_w
```

---

## Scenario 2：Charge Friendly Fire

```text
########
#C..E.P#
########
```

Charger 向东冲锋。

玩家应有机会让 Charger 撞 Enemy。

---

## Scenario 3：Bomber Escape

```text
.....
..B..
..P..
.....
```

Bomber countdown=1。

比较：

```text
move
dash
emp
attack
```

---

## Scenario 4：Greedy Gem Trap

```text
Gem 很近
但路径处于 Charger / Archer 威胁线
```

测试 NanoJev 是否会放弃眼前 gem。

---

## Scenario 5：Sacrifice HP for Combo

玩家可以：

```text
安全撤退
```

或者：

```text
承受 5 HP
换一个环境双杀
```

测试长期收益。

---

# 25. V2 成功判定

V2 不以“画面更丰富”为成功标准。

真正成功标准：

### A. 决策空间扩大

```text
Avg Branch Factor >= 6
```

### B. 死局减少

```text
高怪物密度下 Deadlock Rate 明显低于 V1
```

### C. 策略分化

同一个状态：

```text
Rule
NanoJev
Search
```

出现明显不同策略。

### D. 长期规划有价值

Depth=6 Search 显著优于 greedy / 1-step。

### E. NanoJev 有学习空间

NanoJev 能学到：

```text
诱导
控制
环境利用
放弃短期奖励
```

而不只是：

```text
离 gem 更近
```

---

# 26. 最终目标架构

```text
                   ┌──────────────────┐
                   │   Arena V2 Env   │
                   └────────┬─────────┘
                            │ clone()
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        RuleAgentV2      Beam Search      MCTS
              │             │             │
              └──────┬──────┴──────┬──────┘
                     │             │
                     ▼             ▼
               Dataset Generator
                     │
                     ▼
                  NanoJev
                     │
                     ▼
                Runtime Agent
                     │
                     ▼
               Benchmark Arena
```

最终研究方向：

```text
Search 很聪明，但慢。
NanoJev 很快，但能力有限。

目标：
让 NanoJev 学会 Search 的一部分战术能力，
在本地 GTX 1660 Super 上用较低成本进行实时决策。
```

---

# 27. 最优先 TODO 总表

## P0 — 必做

- [x] 冻结 V1 baseline
- [x] Enemy Intent
- [x] Charger
- [x] Shove
- [x] Enemy 环境伤害
- [x] Bomber
- [x] Friendly Fire
- [x] Dash
- [x] Observation V2
- [x] Candidate V2
- [x] tactical scenario tests
- [x] RuleAgentV2
- [x] Complexity benchmark

## P1 — 强烈建议

- [x] Archer
- [x] 不同敌人移动节奏
- [x] 玩家持久远程武器与有限弹药
- [x] EMP
- [x] Cooldown
- [x] 2 AP
- [x] Barrel
- [x] Map solvability check
- [x] Dataset V2
- [x] NanoJev V2 smoke train

## P2 — 后续

- [ ] Pit
- [ ] Spike
- [ ] Beam Search
- [ ] Search teacher dataset
- [ ] MCTS
- [ ] Search distillation

## P3 — 暂缓

- [ ] Roguelike Build
- [ ] Equipment
- [ ] Skill Draft
- [ ] Procedural tactical rooms
- [ ] Rewind Token
- [ ] 多角色系统

---

# 28. 一句话版本

Jev Arena V2 不应该继续做：

```text
更多怪 → 更难躲
```

而应该做成：

```text
更多怪
↓
更多 Intent
↓
更多碰撞与连锁
↓
更多可利用关系
↓
更多长期策略
↓
NanoJev 真正开始做战术推理
```

**第一阶段最值得投入的四个机制：**

```text
Enemy Intent
+
Shove
+
Charger / Bomber
+
敌我共享环境伤害
```

这四项完成后，再决定是否继续扩大技能、地形和 Search 系统。
