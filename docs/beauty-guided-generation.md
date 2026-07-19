# Beauty-guided param generation — 设计讨论 (brainstorm)

> **状态:探索中 / 未定案** (dumped 2026-07-18)。这是一次思路讨论的记录,供后续慢慢考虑,
> 不是已确定的实现方案。

## 目标

现在有两套模型:

- **face2param**(`predict.py` / `Face2Param`):图片 → 205 维 param,**忠实**重建一个匹配输入脸的
  游戏角色。但"忠实" ≠ "好看":它只保证像,不保证 beauty score 高。
- **beauty_score**(`beauty_score.py`,本次新增):图片 → beauty 分数 [1,5]。**假设**先把它当作
  我们的评分模型(reward model),暂时不纠结 OOD。

**想做的事:** 把两者拼成一条 pipeline,用 reward model 去引导,让 face2param 生成**分数更高**的
param,而不只是贴近人脸特征的 param。本质上是"**在保持像本人的前提下,生成最好看的那版 param**"——
一个 similarity ↔ beauty 的权衡(一个 λ 旋钮)。

经验证据:输入图在图空间被美化后,face2param 的产出确实更漂亮(实测某微调图 4.1 > 原图 3.7,gap 小
但方向对)。

## 核心难点:reward 打分的是「图」,head 吐的是「param」

- beauty_score 的输入是**图片**(图 → DINOv2 → 分数)。
- head 产出的是 **param**。
- 要给 param 打分,必须先**渲染**成角色图(HS2ABMX 写卡 → 出图),再喂 reward。
- **渲染器是不可导黑盒**,梯度无法从 `param → 渲染 → 分数` 倒流回 head。

"冻结全部、只训一层"的思路本身没问题,**卡的就是这条梯度**。

## 三条路线

### A. 可导「代理 reward」(推荐)
离线学一个小网络 `S(param) → beauty`,逼近"渲染+打分"黑盒。beauty 项就可导了,"加一层 head、其余
冻结"直接能训:

```
L = λ·‖refined − base‖²  +  (1−λ)·( −S(refined) )
        保持像本人             把分数往高推
```

- **关键红利:代理训练数据几乎白嫖。** 现有 `data/cards` 就是"已渲染角色图 + 已知 param"的配对;
  拿 beauty_score 把每张卡图打一遍,立刻得到成千上万条 `(param, beauty)`,**不用新渲染**就能训 S。
  (feature/param/beauty 三者在 card 数据集上天然对齐。)
- **风险:reward hacking** —— head 会钻 S 的漏洞,把 param 推到"S 说好看、真渲染出来其实丑"的区域。
  缓解:① trust region(强约束 refined 别离 base 太远);② 迭代一两轮——把 head 新产出真渲染、
  重新打分、再更新 S(即带学习型奖励模型的离线 RL)。

### B. 黑盒 RL / 进化策略
REINFORCE 或 CMA-ES,直接拿"渲染+打分"当黑盒 reward 更新那层 head。不用代理,但**每次评估都要真
渲染**,HS2ABMX 出图秒级 → 几千次 = 几小时到几天。做小 demo 行,规模化太重。**一般不建议。**

### C. 绕开渲染器:图空间先美化,再走现有 face2param(最省事,已有正向证据)

```
输入图 → (在 DINOv2/ArcFace 空间里,约束身份不变地把 beauty 往上推)
       → 更好看的图 → 现有 face2param → 更好看的 param
```

- beauty 和 face2param 都直接吃**图**,梯度在图空间是通的,**没有不可导渲染器问题**,不用新加 head,
  **无 reward-hacking 风险**。
- 代价:改的是"输入长相"而非直接优化 param head;身份保持要靠约束(可复用项目已有的 `arcface`
  身份 backbone 做 identity 相似度约束)。

> **注:** 路线 A/B 都依赖一个能吃 param 的渲染器——这一环的深入调研(逆向游戏渲染 recipe、
> 可导渲染器)见 [offline-renderer-research.md](offline-renderer-research.md)。一个**可导** offline
> 渲染器能让 `param → 渲染 → 图 → beauty` 全程可导,直接溶解本文的核心难点。

## 关于用户提的架构

"**face2param 后面挂一层 MLP、冻结其余、只训这层、平衡 similarity 与 beauty**" —— **成立,就是路线 A
的形态**。唯一要补的,是让 beauty 变可导的**代理 S**;补上它,架构原封不动就能训。

## 一句实话:OOD 会回来找你

reward-hacking 与 OOD 会**叠加**:拿一个"没见过二次元"的 reward 去优化"渲染出来的二次元角色"。
4.1 vs 3.7 的小 gap 很可能落在 OOD 噪声里。机制能跑通,但"**分数更高 = 人眼更好看**"最终取决于 reward
model 在**渲染角色这个域**上到底准不准。真正的 OOD 解法是:在**目标域(游戏/二次元脸)收一批 beauty
标注**去校准 / 微调 reward(靠迁移不靠谱)。相关背景见记忆 `anime-input-failure-mode`。

## 建议 & 开放问题

- **建议先做路线 C**:复用现有一切、已有正向证据、成本低,先确认"图空间美化 → 更漂亮 param"闭环成立。
- **若坚持把 beauty 焊进 param head 本身**,再上路线 A(代理数据几乎白嫖,架构就是设想的样子)。
- 开放问题:
  1. similarity 项怎么定义最稳?param 空间 L2 vs 渲染图的 ArcFace 身份相似度(后者又要渲染)。
  2. λ 取值 / 是否要一条 Pareto 前沿(不同 λ 给用户挑)。
  3. 代理 S 的迭代闭环要跑几轮才收敛、reward-hacking 到什么程度就停。
  4. OOD:要不要先花力气在目标域收标注,再谈优化——否则优化的是一个不可信的信号。
