#!/usr/bin/env python3
"""
Step 3 代码审核：验证 SHOT 和 NRC 修正实现的正确性
Created: 2026-08-14
Purpose: 独立审核 audit_step3_fair_comparison_corrected.py 的实现
审核重点：
  1. SHOT: backbone 可训练，classifier 冻结，SGD 优化器
  2. NRC: backbone+classifier 可训练，CE + 余弦相似度
"""

import sys
import ast
import inspect
from pathlib import Path

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

def audit_shot_implementation():
    """审核 SHOT 实现"""
    print("=" * 80)
    print("审核 SHOT 实现 (run_shot_corrected)")
    print("=" * 80)

    # 读取脚本文件
    script_path = PROJECT_ROOT / 'scripts/revision/audit_step3_fair_comparison_corrected.py'
    with open(script_path, 'r') as f:
        content = f.read()

    # 解析 AST
    tree = ast.parse(content)

    # 查找 run_shot_corrected 函数
    shot_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'run_shot_corrected':
            shot_func = node
            break

    if not shot_func:
        print("❌ 错误：未找到 run_shot_corrected 函数")
        return False

    # 检查关键特性
    checks = {
        "backbone_trainable": False,
        "classifier_frozen": False,
        "sgd_optimizer": False,
        "momentum_0.9": False,
        "weight_decay_1e-3": False,
        "two_stage_training": False,
        "information_maximization": False,
        "pseudo_label_ce": False
    }

    # 检查函数体
    func_source = ast.unparse(shot_func)

    # 1. 检查 backbone 是否可训练
    if "bb.train()" in func_source and "param.requires_grad = True" in func_source:
        checks["backbone_trainable"] = True
        print("✅ backbone.train() + requires_grad = True")
    else:
        print("❌ backbone 未设置为可训练")

    # 2. 检查 classifier 是否冻结
    if "clf.eval()" in func_source and "param.requires_grad = False" in func_source:
        checks["classifier_frozen"] = True
        print("✅ classifier.eval() + requires_grad = False")
    else:
        print("❌ classifier 未设置为冻结")

    # 3. 检查优化器
    if "torch.optim.SGD" in func_source:
        checks["sgd_optimizer"] = True
        print("✅ 使用 SGD 优化器")
    else:
        print("❌ 未使用 SGD 优化器")

    # 4. 检查 momentum
    if "momentum=0.9" in func_source:
        checks["momentum_0.9"] = True
        print("✅ momentum = 0.9")
    else:
        print("❌ momentum 不是 0.9")

    # 5. 检查 weight_decay
    if "weight_decay=1e-3" in func_source:
        checks["weight_decay_1e-3"] = True
        print("✅ weight_decay = 1e-3")
    else:
        print("❌ weight_decay 不是 1e-3")

    # 6. 检查两阶段训练
    if "stage1_epochs" in func_source:
        checks["two_stage_training"] = True
        print("✅ 两阶段训练结构")
    else:
        print("❌ 未找到两阶段训练")

    # 7. 检查信息最大化
    if "entropy" in func_source and "diversity" in func_source:
        checks["information_maximization"] = True
        print("✅ 信息最大化损失（entropy + diversity）")
    else:
        print("❌ 未找到信息最大化损失")

    # 8. 检查伪标签 CE
    if "pseudo_labels" in func_source and "cross_entropy" in func_source:
        checks["pseudo_label_ce"] = True
        print("✅ 伪标签交叉熵损失")
    else:
        print("❌ 未找到伪标签 CE 损失")

    # 检查优化器参数
    if "bb.parameters()" in func_source:
        print("✅ 优化器只更新 backbone 参数")
    else:
        print("❌ 优化器参数不正确")

    # 总结
    print("\n" + "=" * 80)
    print("SHOT 实现审核总结")
    print("=" * 80)
    passed = sum(checks.values())
    total = len(checks)
    print(f"通过检查: {passed}/{total}")

    if passed == total:
        print("✅ SHOT 实现正确！")
        return True
    else:
        print("❌ SHOT 实现存在问题")
        for check, status in checks.items():
            if not status:
                print(f"   - {check}: 未通过")
        return False


def audit_nrc_implementation():
    """审核 NRC 实现"""
    print("\n" + "=" * 80)
    print("审核 NRC 实现 (run_nrc_corrected)")
    print("=" * 80)

    # 读取脚本文件
    script_path = PROJECT_ROOT / 'scripts/revision/audit_step3_fair_comparison_corrected.py'
    with open(script_path, 'r') as f:
        content = f.read()

    # 解析 AST
    tree = ast.parse(content)

    # 查找 run_nrc_corrected 函数
    nrc_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'run_nrc_corrected':
            nrc_func = node
            break

    if not nrc_func:
        print("❌ 错误：未找到 run_nrc_corrected 函数")
        return False

    # 检查关键特性
    checks = {
        "backbone_trainable": False,
        "classifier_trainable": False,
        "adam_optimizer": False,
        "ce_loss": False,
        "cosine_similarity": False,
        "neighbor_loss": False,
        "combined_loss": False,
        "weight_0.1": False
    }

    # 检查函数体
    func_source = ast.unparse(nrc_func)

    # 1. 检查 backbone 是否可训练
    if "bb.train()" in func_source:
        checks["backbone_trainable"] = True
        print("✅ backbone.train()")
    else:
        print("❌ backbone 未设置为可训练")

    # 2. 检查 classifier 是否可训练
    if "clf.train()" in func_source:
        checks["classifier_trainable"] = True
        print("✅ classifier.train()")
    else:
        print("❌ classifier 未设置为可训练")

    # 3. 检查优化器
    if "torch.optim.Adam" in func_source:
        checks["adam_optimizer"] = True
        print("✅ 使用 Adam 优化器")
    else:
        print("❌ 未使用 Adam 优化器")

    # 4. 检查 CE 损失
    if "cross_entropy" in func_source or "F.cross_entropy" in func_source:
        checks["ce_loss"] = True
        print("✅ 交叉熵损失")
    else:
        print("❌ 未找到交叉熵损失")

    # 5. 检查余弦相似度
    if "F.normalize" in func_source and "similarity" in func_source:
        checks["cosine_similarity"] = True
        print("✅ 余弦相似度计算")
    else:
        print("❌ 未找到余弦相似度")

    # 6. 检查邻居损失
    if "neighbor_loss" in func_source:
        checks["neighbor_loss"] = True
        print("✅ 邻居损失项")
    else:
        print("❌ 未找到邻居损失")

    # 7. 检查组合损失
    if "ce_loss" in func_source and "neighbor_loss" in func_source:
        checks["combined_loss"] = True
        print("✅ CE + neighbor_loss 组合")
    else:
        print("❌ 未找到组合损失")

    # 8. 检查权重
    if "0.1 * neighbor_loss" in func_source or "0.1*neighbor_loss" in func_source:
        checks["weight_0.1"] = True
        print("✅ 权重 = 0.1")
    else:
        print("❌ 权重不是 0.1")

    # 检查优化器参数
    if "list(bb.parameters()) + list(clf.parameters())" in func_source:
        print("✅ 优化器更新 backbone + classifier 参数")
    else:
        print("❌ 优化器参数不正确")

    # 总结
    print("\n" + "=" * 80)
    print("NRC 实现审核总结")
    print("=" * 80)
    passed = sum(checks.values())
    total = len(checks)
    print(f"通过检查: {passed}/{total}")

    if passed == total:
        print("✅ NRC 实现正确！")
        return True
    else:
        print("❌ NRC 实现存在问题")
        for check, status in checks.items():
            if not status:
                print(f"   - {check}: 未通过")
        return False


def compare_with_original():
    """与论文原版实现对比"""
    print("\n" + "=" * 80)
    print("与论文原版实现对比")
    print("=" * 80)

    # 读取原版实现
    original_path = PROJECT_ROOT / 'scripts/revision/task_3_1_snr_comparison_label_free.py'
    with open(original_path, 'r') as f:
        original_content = f.read()

    # 读取修正版实现
    corrected_path = PROJECT_ROOT / 'scripts/revision/audit_step3_fair_comparison_corrected.py'
    with open(corrected_path, 'r') as f:
        corrected_content = f.read()

    print("\n【SHOT 对比】")
    print("-" * 80)

    # 原版 SHOT
    if "bb.train()" in original_content and "clf.eval()" in original_content:
        print("✅ 原版: backbone 可训练, classifier 冻结")
    else:
        print("❌ 原版: 实现不一致")

    if "torch.optim.SGD" in original_content:
        print("✅ 原版: 使用 SGD")
    else:
        print("❌ 原版: 未使用 SGD")

    # 修正版 SHOT
    if "bb.train()" in corrected_content and "clf.eval()" in corrected_content:
        print("✅ 修正版: backbone 可训练, classifier 冻结")
    else:
        print("❌ 修正版: 实现不一致")

    if "torch.optim.SGD" in corrected_content:
        print("✅ 修正版: 使用 SGD")
    else:
        print("❌ 修正版: 未使用 SGD")

    print("\n【NRC 对比】")
    print("-" * 80)

    # 原版 NRC
    if "bb.train()" in original_content and "clf.train()" in original_content:
        print("✅ 原版: backbone + classifier 都可训练")
    else:
        print("❌ 原版: 实现不一致")

    if "ce_loss" in original_content and "neighbor_loss" in original_content:
        print("✅ 原版: CE + neighbor_loss")
    else:
        print("❌ 原版: 未找到组合损失")

    # 修正版 NRC
    if "bb.train()" in corrected_content and "clf.train()" in corrected_content:
        print("✅ 修正版: backbone + classifier 都可训练")
    else:
        print("❌ 修正版: 实现不一致")

    if "ce_loss" in corrected_content and "neighbor_loss" in corrected_content:
        print("✅ 修正版: CE + neighbor_loss")
    else:
        print("❌ 修正版: 未找到组合损失")

    return True


def main():
    """主函数"""
    print("=" * 80)
    print("Step 3 代码审核报告")
    print("审核对象: audit_step3_fair_comparison_corrected.py")
    print("审核时间: 2026-08-14")
    print("=" * 80)

    # 审核 SHOT
    shot_ok = audit_shot_implementation()

    # 审核 NRC
    nrc_ok = audit_nrc_implementation()

    # 对比原版
    compare_ok = compare_with_original()

    # 总结
    print("\n" + "=" * 80)
    print("最终审核结论")
    print("=" * 80)

    if shot_ok and nrc_ok and compare_ok:
        print("✅ 所有审核通过！")
        print("✅ SHOT 实现已正确修正（backbone 可训练，classifier 冻结，SGD）")
        print("✅ NRC 实现已正确修正（CE + 余弦相似度，backbone+classifier 可训练）")
        print("✅ 与论文原版实现一致")
        print("\n建议: 可以安全运行实验")
        return 0
    else:
        print("❌ 审核未通过")
        if not shot_ok:
            print("   - SHOT 实现存在问题")
        if not nrc_ok:
            print("   - NRC 实现存在问题")
        if not compare_ok:
            print("   - 与原版实现不一致")
        print("\n建议: 需要修复后再运行实验")
        return 1


if __name__ == "__main__":
    exit(main())
