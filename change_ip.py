import sys
import time
from kortex_api.autogen.client_stubs.NetworkConfigClientRpc import NetworkConfigClient
from kortex_api.autogen.messages import NetworkConfig_pb2
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.TransportClientTcp import TransportClientTcp

def main():
    # 机械臂当前的配置
    CURRENT_IP = "192.168.1.10" 
    USERNAME = "admin"
    PASSWORD = "admin"  # 如果你改过密码，请修改这里

    # 你想要改成的目标新配置（192.168.8.10）
    NEW_IP = "192.168.8.10"
    NEW_SUBNET = "255.255.255.0"
    NEW_GATEWAY = "192.168.8.1"

    # 初始化 TCP 连接 (走 10000 端口，绕过 SSH)
    transport = TransportClientTcp()
    router = RouterClient(transport, lambda error: print(f"Router error: {error}"))
    
    print(f"正在尝试连接机械臂 {CURRENT_IP}...")
    try:
        transport.connect(CURRENT_IP, 10000)
    except Exception as e:
        print(f"连接失败，请检查网线或 IP 是否正确: {e}")
        return

    # 建立会话
    session_info = NetworkConfig_pb2.SessionOpenRequest()
    session_info.username = USERNAME
    session_info.password = PASSWORD
    session_manager = SessionManager(router)
    session_manager.CreateSession(session_info)

    # 创建网络配置客户端
    net_config_client = NetworkConfigClient(router)

    try:
        print("已成功建立 API 会话，正在配置新网络...")
        
        # 1 代表有线网口有线连接
        interface_id = NetworkConfig_pb2.NetworkInterfaceHandle()
        interface_id.interface_handle = 1 

        # 构造新网段的静态 IP
        ip_config = NetworkConfig_pb2.IPv4Configuration()
        ip_config.ip_address = NEW_IP
        ip_config.subnet_mask = NEW_SUBNET
        ip_config.default_gateway = NEW_GATEWAY
        ip_config.dhcp_status = NetworkConfig_pb2.Static

        print(f"正在发送指令：将 IP 修改为 {NEW_IP} ...")
        net_config_client.SetIPv4Configuration(ip_config, interface_id)
        
        print("\n[成功] 指令已发送！机械臂有线网口正在重启并切换至新 IP。")
        print("请注意：当前连接已失效。")

    except Exception as e:
        print(f"修改失败: {e}")
    finally:
        session_manager.CloseSession()
        transport.disconnect()

if __name__ == "__main__":
    main()