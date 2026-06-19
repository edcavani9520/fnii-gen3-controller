import pygame
import os

def test_xbox_controller():
    # 初始化 pygame 的操纵杆模块
    pygame.init()
    pygame.joystick.init()

    # 检查是否连接了手柄
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("❌ 未检测到手柄，请检查接收器是否连好。")
        return

    # 初始化第一个手柄 (Xbox 手柄通常是第一个)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"✅ 已连接手柄: {joystick.get_name()}")
    print("按 'Menu/Start' 键或 'Ctrl+C' 退出程序")
    print("-" * 50)

    clock = pygame.time.Clock()
    keep_running = True

    try:
        while keep_running:
            # 必须处理 pygame 事件，否则数据不会更新
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    keep_running = False

            # 清除屏幕，以便动态显示 (可选)
            # os.system('cls' if os.name == 'nt' else 'clear')

            output = []

            # 1. 读取摇杆 (Axes)
            # Xbox 手柄通常有 6 个轴:
            # Axis 0: 左摇杆左右 (-1 到 1)
            # Axis 1: 左摇杆上下 (-1 到 1)
            # Axis 2: 左扳机 LT (-1 到 1, 初始通常是 -1)
            # Axis 3: 右摇杆左右
            # Axis 4: 右摇杆上下
            # Axis 5: 右扳机 RT
            axes_count = joystick.get_numaxes()
            axis_info = " | ".join([f"A{i}: {joystick.get_axis(i):>6.2f}" for i in range(axes_count)])
            output.append(f"摇杆/扳机 (Axes): {axis_info}")

            # 2. 读取按钮 (Buttons)
            # A: 0, B: 1, X: 2, Y: 3, LB: 4, RB: 5, View: 6, Menu: 7
            buttons_count = joystick.get_numbuttons()
            btn_info = "".join([str(joystick.get_button(i)) for i in range(buttons_count)])
            output.append(f"按钮状态 (Buttons): {btn_info} (A:0, B:1, X:2, Y:3)")

            # 3. 读取十字键 (Hats)
            hats_count = joystick.get_numhats()
            for i in range(hats_count):
                output.append(f"十字键 (Hat {i}): {joystick.get_hat(i)}")

            # 实时刷新显示
            print("\r" + " | ".join(output), end="")

            # 退出机制：按 Xbox 菜单键 (通常是 ID 7)
            if joystick.get_button(7):
                print("\n\n退出测试...")
                break

            # 限制刷新率 (20Hz 足够用于测试且不占 CPU)
            clock.tick(20)

    except KeyboardInterrupt:
        print("\n测试停止。")
    finally:
        pygame.quit()

if __name__ == "__main__":
    test_xbox_controller()

'''
===============================================================================
            XBOX 手柄映射参考文档 (Pygame 标准 / Kinova 机械臂增强版)
===============================================================================

1. 摇杆与扳机 (Axes - 连续数值范围: -1.0 到 1.0)
-------------------------------------------------------------------------------
    编号 [ID]    |  部件名称          | 建议功能 (针对 Kinova Gen3)
    ------------|-------------------|------------------------------------------
    Axis 0      | 左摇杆 - 左右       | 末端 X 轴平移 (左右移动)
    Axis 1      | 左摇杆 - 上下       | 末端 Y 轴平移 (前后移动)
    Axis 2      | 左扳机 (LT)        | 末端 Z 轴下降 (向下压)
    Axis 3      | 右摇杆 - 左右       | 末端 Yaw 偏航角旋转 / 或 Y 轴平移 (取决于习惯)
    Axis 4      | 右摇杆 - 上下       | 末端 Pitch 俯仰角旋转
    Axis 5      | 右扳机 (RT)        | 末端 Z 轴上升 (垂直向上提)

2. 按钮 (Buttons - 离散数值: 按下为 1, 松开为 0)
-------------------------------------------------------------------------------
    编号 [ID]    |  按键名称          | 建议功能 (集成 RL 实验需求)
    ------------|-------------------|------------------------------------------
    Button 0    | A 键               | 夹爪合拢 (逐段关闭或一键全关)
    Button 1    | B 键               | 夹爪张开 (一键释放)
    Button 2    | X 键               | 坐标系切换 (Base 世界坐标 <-> Tool 工具坐标)
    Button 3    | Y 键               | 姿态复位 (回到指定的 Home 初始位置)
    Button 4    | 左肩键 (LB)        | 精细操作模式 (按住时移动速度减半，用于对准)
    Button 5    | 右肩键 (RB)        | 极速模式 (按住时提高移动步长，用于快速位移)
    Button 6    | 查看键 (View)      | 撤销上一步 (Undo) 或 清除当前缓存路点
    Button 7    | 菜单键 (Menu)      | 软件紧急停止 (Stop All Actions)
    Button 8    | 左摇杆中键 (LS)     | 记录当前路点 (Save Waypoint to Dataset)
    Button 9    | 右摇杆中键 (RS)     | 切换控制模式 (Cartesian 笛卡尔 <-> Joint 关节)

3. 十字键 (Hats - 二元组状态)
-------------------------------------------------------------------------------
    编号 [ID]    |  状态输出 (x, y)   | 建议功能
    ------------|-------------------|------------------------------------------
    Hat 0 (上)   | (0, 1)            | 步进式增加 Z 轴高度 (微调)
    Hat 0 (下)   | (0, -1)           | 步进式减少 Z 轴高度 (微调)
    Hat 0 (左)   | (-1, 0)           | 逆时针旋转末端 Roll (翻滚角)
    Hat 0 (右)   | (1, 0)            | 顺时针旋转末端 Roll (翻滚角)

===============================================================================
开发者备忘录:
1. 组合键逻辑: 建议设置 LB + Menu 为“完全退出脚本”，防止误触 Menu 导致实验意外中断。
2. 线性映射: 摇杆数值建议使用立方映射 (Output = Input^3)，这样在小幅度推杆时会更加丝滑。
3. 夹爪保护: A/B 键控制夹爪时，建议在程序中加入当前位置判断，防止电机过载。
===============================================================================
'''