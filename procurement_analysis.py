import os
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.base import LLM
from dashscope import Generation

class QwenLLM(LLM):
    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("DASHSCOPE_API_KEY") or "请设置DASHSCOPE_API_KEY环境变量"

    def call(self, prompt: str, context=None, **kwargs) -> str:
        try:
            prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}" if context else prompt
            response = Generation.call(model='qwen-max', prompt=prompt, api_key=self.api_key)
            return response.output.text if response.status_code == 200 else f"API调用失败: {response.message}"
        except Exception as e:
            return f"调用通义千问API时出错: {str(e)}"

    def type(self) -> str:
        return "qwen"

def main():
    try:
        # 加载数据和初始化
        df = pd.read_csv('Procurement KPI Analysis Dataset.csv')
        smart_df = SmartDataframe(df, config={"llm": QwenLLM(), "verbose": False})
        
        # 定义问题和手动结果
        sections = {
            "三个最难的问题分析结果": [
                "计算每个供应商的总采购金额是多少？",
                "计算每个物品类别的平均单价是多少？",
                "计算所有订单金额的中位数是多少？"
            ],
            "三个趋势分析问题结果": [
                "分析各供应商的采购金额随时间的变化趋势",
                "分析各物品类别的采购数量随时间的变化趋势",
                "分析平均单价随时间的变化趋势"
            ]
        }
        
        # 计算手动结果
        manual_results = [
            df.groupby('Supplier').apply(lambda x: (x['Quantity'] * x['Unit_Price']).sum()).to_dict(),
            df.groupby('Item_Category')['Unit_Price'].mean().to_dict(),
            (df['Quantity'] * df['Unit_Price']).median()
        ]
        
        # 输出结果到文件
        with open('analysis_results.txt', 'w', encoding='utf-8') as f:
            f.write("采购KPI分析结果汇总\n" + "=" * 50 + "\n\n")
            
            for section, questions in sections.items():
                f.write(f"{section}：\n" + "=" * 50 + "\n\n")
                
                for i, q in enumerate(questions, 1):
                    f.write(f"问题 {i}: {q}\n" + "-" * 50 + "\n")
                    f.write(f"AI分析结果: {smart_df.chat(q)}\n")
                    
                    # 只为最难的问题添加手动计算结果
                    if section == "三个最难的问题分析结果" and i <= 3:
                        f.write(f"手动计算结果: {manual_results[i-1]}\n")
                    
                    f.write("\n")
        
        print("分析结果已保存到 analysis_results.txt 文件中")
            
    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main() 