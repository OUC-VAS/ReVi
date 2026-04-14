import torch
import os


def rename_keys_in_checkpoint(
        input_path: str,
        output_path: str,
        key_in_checkpoint: str = None
):
    """
    修改checkpoint中state_dict的键名：
    - 将包含 'BAM' 的子串替换为 'MKE'
    - 将包含 'OEM' 的子串替换为 'SKA'

    Args:
        input_path: 输入权重文件路径
        output_path: 输出权重文件路径
        key_in_checkpoint: 如果checkpoint是一个字典且state_dict保存在某个键下（如'model_state_dict'），则指定该键名；
                          如果checkpoint直接就是state_dict，则设为None
    """
    # 加载checkpoint
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    # 提取state_dict
    if key_in_checkpoint is not None and key_in_checkpoint in checkpoint:
        state_dict = checkpoint[key_in_checkpoint]
        is_nested = True
    else:
        state_dict = checkpoint
        is_nested = False

    # 修改键名
    new_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = old_key.replace('BAM', 'SKA').replace('OEM', 'MKE')
        new_state_dict[new_key] = value

    # 保存
    if is_nested:
        checkpoint[key_in_checkpoint] = new_state_dict
        torch.save(checkpoint, output_path)
    else:
        torch.save(new_state_dict, output_path)

    print(f"修改完成！已保存到: {output_path}")
    print(f"共修改 {len(new_state_dict)} 个参数")


if __name__ == "__main__":
    # 使用示例（请根据实际情况修改路径）
    rename_keys_in_checkpoint(
        input_path="test_weight/pre_pro.pth",
        output_path="pre_pro.pth",
        key_in_checkpoint="model_state_dict"  # 如果checkpoint结构是 {'model_state_dict': {...}}，则填写此键名；否则设为None
    )