# GelSight Mini 夹爪打印结构件 / Printable Gripper Parts

本包包含夹爪底座，以及右侧、左侧三角加强板的两版结构件。
This package contains the gripper base and the right- and left-reinforced bracket variants.

| 结构预览 / Preview | 文件 / File | 内容 / Part | 尺寸 / Size (mm) |
| --- | --- | --- | --- |
| <img src="../fig/gelsight_gripper_base.png" width="200" alt="夹爪底座 / Gripper base"> | [gelsight_gripper_base.stl](../gelsight_gripper_base.stl) | 原始夹爪底座 / Original gripper base | 56 × 16.25 × 24.5 |
| <img src="../fig/gelsight_gripper_right_v9.png" width="200" alt="右侧三角板 V9 / Right-reinforced V9"> | [gelsight_gripper_right_v9.stl](../gelsight_gripper_right_v9.stl) | 右侧三角板结构件 / Right-reinforced bracket | 56 × 74.75 × 28.5 |
| <img src="../fig/gelsight_gripper_left_v10.png" width="200" alt="左侧三角板 V10 / Left-reinforced V10"> | [gelsight_gripper_left_v10.stl](../gelsight_gripper_left_v10.stl) | 左侧三角板结构件 / Left-reinforced bracket | 56 × 74.75 × 28.5 |

## 材料清单 / Bill of materials

以下为本项目装配所需的紧固件清单。
Fasteners specified for this project assembly:

| 配件 / Item | 数量 / Quantity | 用途 / Application |
| --- | --- | --- |
| M3 热熔螺母 / M3 heat-set threaded insert | 2 | 嵌入底座，用于连接 Piper 末端夹爪电机驱动结构 / Embedded in the base to connect it to the Piper end-effector gripper motor drive assembly |
| M2.5 螺丝 / M2.5 screw | 8 | 夹爪连接底座 / Gripper-to-base connection |
| M2 螺丝 / M2 screw | 8 | 底座连接末端 / Base-to-end-effector connection |
| M2 螺丝 / M2 screw | 4 | 固定 GelSight Mini / Securing the GelSight Mini |

打印件见上方文件清单。
Printed parts are listed above.

## 改动记录 / Design changes

1. **第一版部署测试**：直接使用原始结构件进行部署测试。

   **Initial deployment and testing:** used the original structural parts directly for deployment testing.

2. **使用中发现的问题**：长时间使用后，夹爪会出现变形。

   **Observed issue:** the gripper deformed after prolonged use.

3. **加固改版**：针对上述问题，增加连接处加固并加厚结构；本包提供右侧三角加强板 V9 和左侧三角加强板 V10。

   **Reinforcement revision:** added reinforcement at the connections and increased structural thickness in response. This package provides the V9 right-side and V10 left-side reinforcement variants.

## 安装与使用说明 / Assembly and use guide

下图汇总底座安装、传感器安装及使用中观察到的变形。照片用于说明装配位置和改版背景，拍摄时的结构不代表本包 V9/V10 加固版的最终形态。

The photos below show base mounting, sensor installation, and deformation observed during use. They illustrate assembly positions and the motivation for the revisions; the photographed parts do not represent the final V9/V10 reinforced geometry in this package.

![底座与传感器安装、使用中变形四联图 / Four-panel installation and deformation reference](../fig/assembly-guide.png)

1. **底座侧视（图 01）**：对照蓝色底座与 Piper 末端夹爪电机驱动结构的相对位置安装。2 个 M3 热熔螺母嵌入底座，用于该连接位置。

   **Base mounting, side view (01):** use the side view to orient the blue base relative to the Piper end-effector gripper motor drive assembly. The two M3 heat-set inserts are embedded in the base for this connection.

2. **底座正视（图 02）**：对照两侧底座的安装位置与固定孔位。底座连接末端使用 M2 螺丝共 8 颗；夹爪结构件连接底座使用 M2.5 螺丝共 8 颗。

   **Base mounting, front view (02):** use the front view to locate both bases and their mounting holes. Use eight M2 screws for the base-to-end-effector connections and eight M2.5 screws for the gripper-to-base connections.

3. **传感器安装（图 03）**：右侧蓝色支架展示 GelSight Mini 的安装位置。使用 M2 螺丝共 4 颗固定 GelSight Mini；装配后检查传感器固定情况及线缆是否妨碍夹爪开合。

   **Sensor installation (03):** the blue bracket on the right shows the GelSight Mini mounting position. Use four M2 screws in total to secure the GelSight Mini sensors. After assembly, check that the sensors are secure and the cables do not obstruct gripper motion.

4. **使用观察（图 04）**：图中为使用过程中观察到的夹爪变形，也是增加连接加固与结构加厚的原因。使用时留意支架变形及紧固件松动，出现异常时停止使用并检查。

   **Inspection during use (04):** the photo shows gripper deformation observed during use, which motivated the reinforced connections and thicker structure. Check for bracket deformation and loose fasteners; stop and inspect the assembly if either occurs.

## 致谢 / Acknowledgements

感谢 [GeWu-Lab](https://github.com/GeWu-Lab) 的 [AnyTouch 2](https://github.com/GeWu-Lab/AnyTouch2) 工作及作者团队提供原始结构件，为本项目的部署测试和结构改进提供了基础。本项目在原始设计上增加了连接处加固与结构加厚。

We thank the authors of [AnyTouch 2](https://github.com/GeWu-Lab/AnyTouch2) from [GeWu-Lab](https://github.com/GeWu-Lab) for sharing the original structural parts that formed the basis of our deployment tests and design revisions. Our modifications add reinforcement at the connections and increase structural thickness.

## 许可证 / License

本项目的结构加固改动及随附文档沿用 SimpleTouch 的 **Apache License 2.0**，详见 [LICENSE](LICENSE)。原始结构件保留原作者的权利及适用许可；本声明不重新许可第三方原始材料。

The structural reinforcement modifications and accompanying documentation in this project follow SimpleTouch's **Apache License 2.0**; see [LICENSE](LICENSE). The original structural parts retain their authors' rights and applicable license terms. This notice does not relicense third-party original materials.
