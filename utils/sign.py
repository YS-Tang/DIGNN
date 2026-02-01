import os
import torch

ENV_DEVICE = os.environ.get('DIGNN_ENV') or os.environ.get('DEVICE') or 'cpu'

class Sign:
    def __init__(self, edge_index_norm, joint, edge_ids_norm=None):
        # 设备自动检测, 尽量使用GPU计算
        self.input_device = edge_index_norm.device
        self.calc_device = ENV_DEVICE
        
        self.edge_index_norm = edge_index_norm.to(self.calc_device)
        self.joint = joint.to(self.calc_device)
        
        if edge_ids_norm is None:
            edge_ids_norm = torch.arange(self.edge_index_norm.shape[1], device=self.calc_device)
        else:
            edge_ids_norm = edge_ids_norm.to(self.calc_device)
        
        
        # 创建索引映射张量 (替代id2idx字典)
        max_eid = edge_ids_norm.max().item() + 1
        self.idx_map = torch.full((max_eid,), -1, dtype=torch.long, device=self.calc_device)
        self.idx_map[edge_ids_norm] = torch.arange(edge_ids_norm.size(0), device=self.calc_device)
        
        # 向量化索引转换 (替代列表推导)
        src_indices = self.idx_map[joint[0]]
        dst_indices = self.idx_map[joint[1]]
        
        # 数据提取
        self.src_edges = self.edge_index_norm[:, src_indices]
        self.dst_edges = self.edge_index_norm[:, dst_indices]
        
        # 分解端点
        self.src_start, self.src_end = self.src_edges[0], self.src_edges[1]
        self.dst_start, self.dst_end = self.dst_edges[0], self.dst_edges[1]
        
        self.check()
    
    
    def check(self):
        # 计算四个可能的连接点
        connection_mask = (
            (self.src_start == self.dst_start) |
            (self.src_start == self.dst_end) |
            (self.src_end == self.dst_start) |
            (self.src_end == self.dst_end)
        )
        
        if not torch.all(connection_mask):
            # GPU兼容的异常处理
            invalid_indices = torch.where(~connection_mask)[0].cpu().numpy()  # 移回CPU处理
            invalid_pairs = self.joint[:, invalid_indices].T.cpu().numpy()
            raise ValueError(f"无效边对索引: {invalid_pairs.tolist()}")
    
    
    def AlignmentSign(self):
        """
        检测对齐. 例如输入数据对index[(1,2),(2,3)], 默认数据对的id为[0,1]
        检测关系(0,1), 发现相同数据2位置错位, 则输出-1
        """
        condition1 = self.src_start == self.dst_start  # 前边头 == 后边头
        condition2 = self.src_end == self.dst_end      # 前边尾 == 后边尾
        
        sign = torch.where(condition1 | condition2, 1, -1).long()
        return sign.to(self.input_device)
    
    
    def PositionSign(self, calc_mode='all'):
        """
        检测相同数据的位置, 如果位于前面则返回1, 位于后面返回-1
        clac_mode接受'former','latter','all'. 分别返回前者, 后者和全部
        
        例如输入数据对index[(1,2),(2,3)], 默认数据对的id为[0,1]
        检测关系(0,1), 发现相同数据2位于前者的后面, 位于后者的前面
        如果要求'former', 则返回-1, 如果要求'latter', 则返回1, 如果要求'all', 则返回(-1,1)
        """
        condition1 = (self.src_start == self.dst_start) | (self.src_start == self.dst_end)
        condition2 = (self.src_start == self.dst_start) | (self.src_end == self.dst_start)
        
        sign1 = torch.where(condition1, 1, -1).long()
        sign2 = torch.where(condition2, 1, -1).long()
        
        return {
            'former': sign1.to(self.input_device),
            'latter': sign2.to(self.input_device),
            'all': (sign1.to(self.input_device), sign2.to(self.input_device))
        }.get(calc_mode, ValueError("无效calc_mode"))
        
        
def Calc_Sign(Pre_AliSign, Pre_PosSignFm, Post_PosSign):
    return Pre_AliSign * Pre_PosSignFm * Post_PosSign ** ((1-Pre_AliSign)/2)