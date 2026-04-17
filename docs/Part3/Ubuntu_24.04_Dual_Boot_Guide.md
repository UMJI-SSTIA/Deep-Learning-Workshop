# Windows + Ubuntu 24.04 双系统安装指南

> 本指南适用于已安装 Windows 10/11 的电脑，在不影响 Windows 系统的前提下安装 Ubuntu 24.04 LTS。
>
> 基于 UEFI + GPT 模式，已在联想拯救者 Y9000P 上验证通过。
> 
> Edited by LKM from SSTIA

---

> ⚠️ **安装双系统涉及磁盘分区操作，存在数据丢失风险。** 请务必：
>
> 1. **在开始之前备份所有重要数据**（文档、照片、代码等）到外部存储设备
> 2. **严格按照本指南的顺序操作**，不要跳步
> 3. 遇到不确定的步骤，先停下来查资料或询问有经验的同学，**不要猜测操作**
>
> 如果操作得当，安装双系统**不会影响你现有的 Windows 系统和数据**。

> 💡 **遇到问题？** 把完整报错信息贴给 ChatGPT/Claude 等 LLM，或在 CSDN、知乎搜索关键词，通常都能找到解决方案。

---

## 目录

- [第一部分：准备工作（在 Windows 中完成）](#第一部分准备工作在-windows-中完成)
  - [1. 确认 BIOS 模式为 UEFI](#1-确认-bios-模式为-uefi)
  - [2. 确认磁盘分区格式为 GPT](#2-确认磁盘分区格式为-gpt)
  - [3. 关闭 BitLocker 加密](#3-关闭-bitlocker-加密)
  - [4. 关闭 Windows 快速启动](#4-关闭-windows-快速启动)
  - [5. 为 Ubuntu 腾出磁盘空间](#5-为-ubuntu-腾出磁盘空间)
  - [6. 下载 Ubuntu 24.04 镜像](#6-下载-ubuntu-2404-镜像)
  - [7. 制作 USB 启动盘](#7-制作-usb-启动盘)
- [第二部分：BIOS 设置](#第二部分bios-设置)
- [第三部分：安装 Ubuntu](#第三部分安装-ubuntu)
- [第四部分：安装后配置](#第四部分安装后配置)
  - [1. 修复双系统时间不同步](#1-修复双系统时间不同步)
  - [2. 设置默认启动系统](#2-设置默认启动系统)
  - [3. 连接 WiFi ，更换软件源（可选），下载常见软件](#3-连接-wifi-更换软件源可选下载常见软件)
- [附录：常见问题与故障排除](#附录常见问题与故障排除)

---

## 第一部分：准备工作（在 Windows 中完成）

> 💡 这部分所有操作都在你现有的 Windows 系统中完成。请按顺序逐步执行。

### 你需要准备

- 一个 **8GB 以上**的 U 盘（制作启动盘时 U 盘会被格式化，请先备份 U 盘中的数据）
- 稳定的网络连接
- 至少 **60GB** 的可用磁盘空间（建议 100GB 以上，尤其是后续要装 CUDA 等深度学习工具）

---

### 1. 确认 BIOS 模式为 UEFI

UEFI 是现代电脑的引导方式，双系统安装需要它。近几年的绝大多数电脑都是 UEFI 模式。

**操作步骤：**

1. 按 `Win + R`，输入 `msinfo32`，回车
2. 在"系统摘要"中找到 **BIOS 模式**，确认其值为 **UEFI**

如果显示的是 **传统（Legacy）**，你需要进入 BIOS 将引导模式改为 UEFI。不同品牌进入 BIOS 的按键不同：

| 品牌 | 进入 BIOS 的按键 |
|------|-----------------|
| 联想（Lenovo/ThinkPad） | F2 或 回车后按 F1 |
| 戴尔（Dell） | F2 |
| 惠普（HP） | F10 |
| 华硕（ASUS） | F2 或 Del |
| 微星（MSI） | Del |
| 宏碁（Acer） | F2 |

> 以上按键需要在**开机出现品牌 Logo 时连续快按**。如果不确定，可以搜索"你的电脑型号 + 进入 BIOS"。

进入 BIOS 后，找到 Boot 或 Startup 相关选项，将引导模式改为 UEFI。**修改后记得保存并退出**（通常是按 F10）。

---

### 2. 确认磁盘分区格式为 GPT

GPT 是与 UEFI 配套的磁盘分区格式。如果不是 GPT，安装 Ubuntu 引导器时会失败。

**操作步骤：**

1. 右键点击"开始"按钮 → 选择 **磁盘管理**
2. 在窗口底部找到 **Windows 所在的磁盘**（通常是"磁盘 0"）
3. **右键点击最左侧的灰色磁盘标签**（写着"磁盘 0"的地方）→ 选择 **属性**
4. 切换到 **卷** 选项卡，查看 **磁盘分区形式**

如果显示 **GUID 分区表 (GPT)**，则没问题，继续下一步。

如果显示 **主启动记录 (MBR)**，需要转换为 GPT。这个操作较复杂，建议搜索"MBR 转 GPT 无损"寻找教程，或使用 Windows 自带的 `mbr2gpt` 工具。

---

### 3. 关闭 BitLocker 加密

BitLocker 是 Windows 的磁盘加密功能。如果不关闭，安装双系统后每次进入 Windows 都可能要求输入恢复密钥，非常麻烦。

**操作步骤：**

1. 打开 **磁盘管理**，查看各分区是否标注了 **"BitLocker 已加密"**
2. 如果有：
   - 打开 **设置 → 隐私和安全性 → 设备加密**，关闭开关
   - 或者搜索 **"BitLocker"**，在控制面板中点击 **"关闭 BitLocker"**
3. 如果没有 BitLocker 标识，跳过此步

> ⚠️ 关闭 BitLocker 解密过程可能需要较长时间（取决于磁盘数据量），建议接通电源后操作。

---

### 4. 关闭 Windows 快速启动

Windows 的快速启动功能会在关机时将部分系统状态保存到磁盘，这可能导致 Ubuntu 无法正常访问 Windows 分区，甚至引发数据损坏。

**操作步骤：**

1. 打开 **控制面板 → 电源选项**（或搜索"电源选项"）
2. 点击左侧 **"选择电源按钮的功能"**
3. 点击上方 **"更改当前不可用的设置"**
4. **取消勾选** "启用快速启动（推荐）"
5. 点击 **保存修改**

---

### 5. 为 Ubuntu 腾出磁盘空间

我们需要从现有分区中压缩出一块"未分配空间"给 Ubuntu 使用。

**操作步骤：**

1. 右键"开始"→ **磁盘管理**
2. 选择一个**空间充裕的分区**（建议选非 C 盘的分区，减少对 Windows 系统分区的影响）
3. **右键该分区** → **压缩卷**
4. 在"输入压缩空间量"中输入要分给 Ubuntu 的大小（单位为 MB）：
   - **最少 60000 MB**（约 60GB）
   - **建议 100000-200000 MB**（100-200GB），尤其是后续要安装 CUDA、PyTorch、数据集等
5. 点击 **压缩**

压缩完成后，磁盘管理中会出现一块 **"未分配"** 的黑色区域。这就是 Ubuntu 将要安装的位置。**不要对这块未分配空间做任何操作**（不要新建卷、不要格式化）。

> ⚠️ 如果可压缩的空间远小于实际剩余空间，可能是因为磁盘碎片或不可移动的系统文件。可以尝试先对该分区进行碎片整理，或使用 **DiskGenius** 等第三方工具压缩。

---

### 6. 下载 Ubuntu 24.04 镜像

下载 Ubuntu 桌面版的 ISO 镜像文件。

**推荐下载地址（国内镜像，速度快）：**

- 清华大学镜像：https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/24.04/
- 中国科技大学镜像：https://mirrors.ustc.edu.cn/ubuntu-releases/24.04/

选择文件名为 **`ubuntu-24.04.x-desktop-amd64.iso`** 的文件下载（其中 `x` 是小版本号，选最新的即可）。

> 注意选 **desktop** 版本（带图形界面），不要选 server 版本。`amd64` 适用于所有 x86_64 架构的电脑（包括 Intel 和 AMD 处理器）。

---

### 7. 制作 USB 启动盘

将下载的 ISO 镜像写入 U 盘，使其成为可引导的安装盘。推荐使用 **Ventoy** 或 **Rufus**。

#### 方式 A：使用 Ventoy（推荐，更灵活）

Ventoy 的优势是**制作一次后，直接把 ISO 文件复制到 U 盘即可**，U 盘剩余空间还能正常存放其他文件。

1. 前往 https://www.ventoy.net/cn/download.html 下载 Windows 版本
2. 解压后运行 **Ventoy2Disk.exe**
3. 在"设备"中确认选中的是你的 U 盘（**请务必确认，选错会格式化其他磁盘！**）
4. 点击 **配置选项 → 分区类型 → 选择 GPT**
5. 点击 **安装**，确认两次格式化提示
6. 安装完成后，将下载的 **`ubuntu-24.04.x-desktop-amd64.iso`** 直接复制到 U 盘根目录

#### 方式 B：使用 Rufus

1. 前往 https://rufus.ie/zh/ 下载最新版本
2. 插入 U 盘，运行 Rufus
3. **设备**：选择你的 U 盘
4. **引导类型选择**：点击"选择"，找到下载的 ISO 文件
5. **分区类型**：选择 **GPT**
6. **目标系统类型**：选择 **UEFI（非CSM）**
7. 其他保持默认，点击 **开始**
8. 弹出写入模式选择时，选择 **"以 ISO 镜像模式写入"**

---

## 第二部分：BIOS 设置

制作好启动 U 盘后，需要修改 BIOS 设置才能从 U 盘启动安装程序。

**操作步骤：**

1. **U 盘保持插入**，重启电脑
2. 在开机出现品牌 Logo 时**连续快按对应按键**进入 BIOS（按键参见第 1 步的表格）
3. 在 BIOS 中进行以下设置：

**① 关闭安全启动（Secure Boot）**

在 Security 或 Boot 菜单中找到 **Secure Boot**，将其设为 **Disabled**。如果不关闭，可能出现 `Verification failed` 错误无法启动安装程序。

> 安装完 Ubuntu 后可以重新开启 Secure Boot。

**② 关闭 Intel RST（如果有）**

如果你的 BIOS 中有 **SATA Controller Mode** 或类似选项，并且当前值为 **Intel RST (RAID)**，将其改为 **AHCI**。如果找不到此选项或已经是 AHCI，则跳过。

> ⚠️ 在已安装 Windows 的情况下将 RST 改为 AHCI 可能导致 Windows 蓝屏。如果遇到，可以搜索"Windows RST 改 AHCI 安全模式"找到解决方案，或者改回 RST 后先在 Windows 中做好准备再切换。

**③ 关闭独显直连（笔记本用户，如果有的话）**

部分游戏本有"独显直连"或"混合模式"开关。安装 Ubuntu 时建议**先切换到混合模式（Hybrid）或关闭独显直连**，避免安装过程中黑屏。装好系统和显卡驱动后再开启。

**④ 设置 U 盘为第一启动项**

在 Boot 菜单中，将 USB 设备移到启动顺序的**最前面**。不同 BIOS 界面操作方式不同，通常可以用 F5/F6 上下移动，或直接拖拽。

4. **保存并退出**（通常按 F10，选择 Save & Exit）

> **如果设置了 U 盘优先启动后仍直接进入 Windows**：
>
> 在 Windows 中：设置 → 系统 → 恢复 → 高级启动 → 点击"立即重新启动"→ 选择"使用设备"→ 选择你的 U 盘。

---

## 第三部分：安装 Ubuntu

从 U 盘启动后，你会进入 Ubuntu 安装界面。

### 3.1 启动安装程序

- 如果使用 **Ventoy**：在 Ventoy 菜单中选择 `ubuntu-24.04.x-desktop-amd64.iso`，然后选择 **Boot in normal mode**
- 如果使用 **Rufus**：直接进入 Ubuntu 启动菜单

选择 **Try or Install Ubuntu**，等待加载。

> 如果在加载过程中遇到黑屏或花屏（尤其是 NVIDIA 显卡），可以在启动菜单中按 `e` 编辑引导参数，找到 `quiet splash` 并在后面加上 `nomodeset`，然后按 `F10` 启动。这会使用基本显示驱动，避免显卡兼容性问题。

### 3.2 安装设置

1. **选择语言**：选择 **简体中文**（或你喜欢的语言）
2. **可访问性**：默认即可，直接"下一步"
3. **键盘布局**：默认即可
4. **网络连接**：可以先跳过，安装后再连接（连接网络会导致安装过程中下载更新，拖慢安装速度）
5. **安装选项**：
   - 选择 **正常安装**（Normal Installation）
   - **取消勾选** "安装 Ubuntu 时下载更新"（加快安装速度，之后手动更新）
   - **勾选** "为图形或无线硬件，以及其他媒体格式安装第三方软件"

### 3.3 分区设置（关键步骤）

安装类型这一步有两种选择：

#### 简易方式（推荐新手）

选择 **"与 Windows Boot Manager 共存安装 Ubuntu"**（Install Ubuntu alongside Windows Boot Manager）。

安装程序会自动使用你之前压缩出的未分配空间。你只需要拖动滑块确认分配给 Ubuntu 的空间大小。确认后点击 **"现在安装"**。

#### 手动分区方式（进阶用户）

选择 **"其他选项"**（Something else），进入手动分区界面。

找到你之前压缩出的那块 **"空闲空间"**（free space），然后依次创建以下分区：

| 分区 | 大小 | 类型 | 挂载点 | 说明 |
|------|------|------|--------|------|
| EFI 系统分区 | 512 MB | EFI System Partition | — | Ubuntu 的引导分区 |
| 交换分区 | 与内存相同（如 16GB 则 16384 MB） | swap | — | 虚拟内存，休眠时使用 |
| 根分区 | 剩余全部空间 | ext4 | `/` | Ubuntu 系统和所有文件 |

> ⚠️ **安装启动引导器的设备**：在分区界面下方的下拉菜单中，选择 **你刚创建的 EFI 分区**对应的设备号（如 `/dev/nvme0n1p5`），或者选择包含 **Windows Boot Manager** 的那个 EFI 分区。**千万不要选错成 Windows 的系统分区。**

确认分区方案无误后，点击 **"现在安装"**，确认弹出的分区修改提示。

### 3.4 完成安装

1. **时区**：选择 **Shanghai**
2. **用户信息**：设置用户名和密码。**用户名建议使用简短的英文**（如 `lkm`），避免使用中文或空格，否则后续使用中可能遇到路径问题
3. 等待安装完成（通常 10-20 分钟）
4. 安装完成后，点击 **"现在重启"**
5. 屏幕提示 **"Please remove the installation medium, then press ENTER"** 时，**拔出 U 盘**，然后按回车

---

## 第四部分：安装后配置

重启后，你应该能看到 **GRUB 引导菜单**，可以选择进入 Ubuntu 或 Windows。

> 如果重启后直接进入了 Windows 而没有 GRUB 菜单，请重新进入 BIOS，将 **Ubuntu** 移到启动顺序的第一位。

### 1. 修复双系统时间不同步

安装双系统后，你可能发现 Windows 和 Ubuntu 的时间不一致。这是因为两个系统对硬件时钟的理解不同：Windows 认为硬件时钟是本地时间，Ubuntu 默认认为是 UTC 时间。

在 Ubuntu 终端中执行（按 `Ctrl + Alt + T` 打开终端）：

```bash
sudo apt update
sudo apt install ntpdate -y
sudo ntpdate time.windows.com
sudo hwclock --localtime --systohc
```

这会让 Ubuntu 也把硬件时钟当作本地时间，与 Windows 保持一致。

### 2. 设置默认启动系统

默认情况下，GRUB 菜单会在 Ubuntu 上等待 10 秒。如果你平时主要用 Windows，可以把默认启动项改为 Windows。

```bash
sudo nano /etc/default/grub
```

找到 `GRUB_DEFAULT=0`，修改为：

```
GRUB_DEFAULT=2
```

> 这里的数字是 GRUB 菜单中的条目索引（从 0 开始）。Ubuntu 通常是 0，Windows 通常是 2。请根据你实际的 GRUB 菜单顺序确定。

保存（`Ctrl + O`，回车）并退出（`Ctrl + X`），然后更新 GRUB：

```bash
sudo update-grub
```

### 3. 连接 WiFi ，更换软件源（可选），下载常见软件

**连接 WiFi**：点击 Ubuntu 桌面右上角的系统托盘区域，选择 WiFi 网络并输入密码。

**更换软件源为国内镜像**（可选）：

```bash
sudo cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak
sudo sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources
sudo sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources
sudo apt update
```

> Ubuntu 24.04 使用了新的 `sources.list.d/ubuntu.sources` 文件格式（DEB822），与旧版的 `sources.list` 不同。上述命令已适配新格式。如果你的系统使用的是旧格式，请搜索"Ubuntu 24.04 更换源"查找对应方法。

**下载安装常用软件**：微信、飞书、Clash Verge等常用软件都有其 Ubuntu Linux 版本，你可以在浏览器中访问软件官网或者 GitHub，下载对应的 .deb 文件，在终端中运行`sudo dpkg -i <文件名称>.deb`完成安装。

---

## 附录：常见问题与故障排除

### Q1：重启后没有 GRUB 菜单，直接进入 Windows

进入 BIOS，将 Ubuntu 设为第一启动项。如果启动项中没有 Ubuntu，说明引导器安装失败，需要用 U 盘重新进入 Ubuntu 安装环境，选择"试用 Ubuntu"，然后使用 `boot-repair` 工具修复：

```bash
sudo add-apt-repository ppa:yannubuntu/boot-repair -y
sudo apt update
sudo apt install boot-repair -y
boot-repair
```

选择 **"推荐修复"** 即可。

### Q2：安装过程中黑屏或花屏

在 GRUB 启动菜单中按 `e`，找到包含 `quiet splash` 的行，在后面加上 `nomodeset`，然后按 `F10` 启动。这将使用兼容的显示驱动。

安装完成进入系统后，如果仍然黑屏，可以在 GRUB 菜单中同样添加 `nomodeset` 参数启动，然后安装 NVIDIA 驱动来彻底解决。

### Q3：进入 Windows 时弹出 BitLocker 恢复密钥

说明你在安装双系统前没有关闭 BitLocker。登录 https://account.microsoft.com/devices/recoverykey ，用你的微软账号找到恢复密钥，输入后即可进入系统。进入后请关闭 BitLocker。

### Q4：双系统时间不一致

参见 [第四部分第 1 步](#1-修复双系统时间不同步)。

### Q5：WiFi 无法使用

部分较新的无线网卡可能需要额外驱动。先用有线网络（USB 网线转接器）或手机 USB 共享网络连接，然后执行：

```bash
sudo apt update
sudo apt install linux-firmware
sudo reboot
```

如果仍不行，搜索你的无线网卡型号（可通过 `lspci | grep -i network` 查看）加"Ubuntu 驱动"关键词。

### Q6：想删除 Ubuntu 恢复为纯 Windows

1. 在 Windows 中使用 **磁盘管理** 删除 Ubuntu 所在的分区（右键 → 删除卷），然后将空间合并回原有分区
2. 修复 Windows 引导：以管理员身份运行 CMD，执行：
   ```
   bcdedit /set {bootmgr} path \EFI\Microsoft\Boot\bootmgfw.efi
   ```
3. 重启后 GRUB 菜单消失，直接进入 Windows

### Q7：BIOS 中切换 RST 到 AHCI 后 Windows 蓝屏

不要慌张。重启进 BIOS 先改回 RST，正常进入 Windows 后：

1. 按 `Win + R`，输入 `msconfig`，回车
2. 切换到 **引导** 选项卡，勾选 **安全引导**，点确定
3. 重启进入 BIOS，将 RST 改为 **AHCI**
4. Windows 会以安全模式启动（这次不会蓝屏）
5. 再次运行 `msconfig`，**取消勾选** 安全引导，确定
6. 正常重启，AHCI 模式下的 Windows 即可正常使用
