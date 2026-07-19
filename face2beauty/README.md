# face2beauty

## 人脸美观度 / 吸引力评分数据集

| Dataset | 大致内容 | Link |
| --- | --- | --- |
| SCUT-FBP5500 | 5,500 张正脸，覆盖 Asian/Caucasian、male/female，带 1–5 beauty score、score distribution 及 landmarks；最标准的 facial beauty prediction benchmark | [GitHub](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) |
| SCUT-FBP | 500 张 Asian female 正脸肖像，带 attractiveness ratings；规模小，但对“亚洲女性脸”的 prior 直接，可作轻量 baseline | [HCII-Lab](http://www.hcii-lab.net/data/SCUT-FBP/EN/introduce.html) |
| MEBeauty | 2,550 张 in-the-wild 多族裔男女脸，约 300 名不同背景打分者，提供 generic/personal scores；比 SCUT 更接近真实照片、非 studio/frontal | [GitHub](https://github.com/fbplab/MEBeauty-database) |
| LiveBeauty | 10,000 张直播场景 face images，约 200,000 条 attractiveness annotations；最接近现代美颜/网感 portrait prettiness（可用性需实际核对） | [GitHub](https://github.com/Estella-LH/FPEM) |
| Chicago Face Database (CFD) | 标准化多族裔人脸照片，带 physical attributes 与 subjective ratings（含 attractiveness）；干净标准化，偏心理学刺激库 | [chicagofaces.org](https://www.chicagofaces.org/) |
| Face Research Lab London Set | 2,513 人的 neutral frontal faces，年龄 17–90，带 1–7 attractiveness ratings；适合做 attractiveness prior，风格偏实验室/心理学 | [figshare](https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666) |

## SCUT-FBP5500 下载链接

已 clone 到 `SCUT-FBP5500-Database-Release/`（仅 splits/脚本/网络定义，已 gitignore）。图片和权重需从下方网盘另行下载。

| 内容 | 大小 | 百度网盘 | Google Drive |
| --- | --- | --- | --- |
| 人脸图 + 86 landmarks | 172MB | [pan.baidu](https://pan.baidu.com/s/1Ff2W2VLJ1ZbWSeV5JbF0Iw)（密码 `if7p`） | [drive](https://drive.google.com/open?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf) |
| Caffe 权重 | 322MB | [pan.baidu](https://pan.baidu.com/s/1byWe21ATKnpGarKY5feg1g)（密码 `owgm`，解压 `12345`） | [drive](https://drive.google.com/file/d/1un5CjTz_49Lg6MTNQn99WD7FjFqEJGoY/view)（解压 `12345`） |
| Pytorch 权重 | 101MB | [pan.baidu](https://pan.baidu.com/s/1OhyJsCMfAdeo8kIZd29yAw)（密码 `ateu`） | — |

> 图片下载后解压到 `SCUT-FBP5500-Database-Release/data/faces/`（`forward.py` 里 `root = '../data/faces'`）。数据集仅限非商业研究用途。
