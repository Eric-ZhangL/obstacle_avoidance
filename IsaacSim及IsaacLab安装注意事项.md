# Isaac Sim 及 Isaac Lab 安装注意事项

本文只记录安装时容易出错的关键点。完整安装步骤请参考对应版本的官方文档。

## 1. 安装前确认

- Isaac Sim、Isaac Lab 和 Python 版本必须相互兼容，不要混用不同版本。
- 确认 Linux 系统、显卡和 NVIDIA 驱动满足 Isaac Sim 要求。
- 使用以下命令检查 GPU 和驱动：

```bash
nvidia-smi
```

## 2. Isaac Sim 安装

Isaac Sim 压缩包解压后可以直接使用，但必须执行一次安装后配置：

```bash
cd /path/to/isaacsim
./post_install.sh
```

使用 Selector 验证本体能否正常启动：

```bash
./isaac-sim.selector.sh
```

若无法启动，优先检查显卡驱动、系统依赖和安装路径。

## 3. Isaac Lab 与 Isaac Sim 连接

设置环境变量，路径必须与实际 Isaac Sim 安装位置一致：

```bash
export ISAACSIM_PATH="/path/to/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

在 Isaac Lab 根目录创建符号链接：

```bash
cd /path/to/IsaacLab
ln -s "${ISAACSIM_PATH}" _isaac_sim
ls -l _isaac_sim
```

如果 `_isaac_sim` 已存在但指向错误，应先确认并重新创建正确链接。

## 4. Conda 环境与安装

在 Isaac Lab 根目录创建并激活环境：

```bash
./isaaclab.sh --conda isaaclab
conda activate isaaclab
```

安装基础依赖和 Isaac Lab 扩展：

```bash
sudo apt update
sudo apt install cmake build-essential
./isaaclab.sh --install
```

每次运行 Isaac Lab 项目前，都要先激活环境：

```bash
conda activate isaaclab
```

检查当前 Python 是否属于该环境：

```bash
which python
```

## 5. 安装验证

先运行最简单的仿真示例：

```bash
python scripts/tutorials/00_sim/create_empty.py
```

再运行无界面训练示例：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-v0 \
  --headless
```

基础示例未通过时，不要直接调试自定义项目。

## 6. 自定义项目安装

在 Isaac Lab 根目录使用 Template Generator 创建新项目：

```bash
cd /path/to/IsaacLab
./isaaclab.sh --new
```

Template Generator 可以选择多种默认环境模板，并根据交互提示生成项目结构。项目生成后，
再根据实际需求修改环境配置、环境逻辑、模型和训练参数。

新建或移动 Isaac Lab 项目后，需要重新执行 editable 安装：

```bash
python -m pip install -e /path/to/project/source/package_name
```

检查 Python 实际导入的包路径，避免加载旧项目：

```bash
python -c "import package_name; print(package_name.__file__)"
```

## 7. VS Code 配置

`.vscode/settings.json` 中的 `python.analysis.extraPaths` 用于代码补全和跳转，例如：

```json
{
    "python.analysis.extraPaths": [
        "/path/to/IsaacLab/source/isaaclab",
        "/path/to/IsaacLab/source/isaaclab_tasks",
        "/path/to/IsaacLab/source/isaaclab_assets",
        "/path/to/IsaacLab/source/isaaclab_rl"
    ]
}
```

该配置只影响 VS Code，不替代 Conda 环境激活、符号链接或 editable 安装。

## 快速排查顺序

出现启动、导入或任务注册错误时，依次检查：

1. `nvidia-smi` 是否正常。
2. Isaac Sim 是否可以单独启动。
3. `_isaac_sim` 是否指向正确目录。
4. 是否已执行 `conda activate isaaclab`。
5. `which python` 是否指向正确 Conda 环境。
6. 自定义任务包的实际导入路径是否正确。
