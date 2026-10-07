# Beyond Image Likeness

现在的pipeline:

```
image -> parameter
```

现在的param已经能够faithfully recover image details了。

但是合成出来的游戏人物建模不够美观，因此需要进行游戏化的tuning。

大致的逻辑是这样：

```
photo likeness (done) → game-style projection → aesthetic refinement
```