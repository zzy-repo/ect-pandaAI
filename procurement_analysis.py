import os
import pandas as pd
import logging
from pandasai import SmartDataframe
from pandasai.llm.base import LLM
from dashscope import Generation

# 配置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenLLM(LLM):
    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

    def call(self, prompt: str, context=None, **kwargs) -> str:
        try:
            if context:
                prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
            
            response = Generation.call(
                model='qwen-max',
                prompt=prompt,
                api_key=self.api_key
            )
            
            if response.status_code == 200:
                return response.output.text
            raise Exception(f"API调用失败: {response.message}")
        except Exception as e:
            logger.error(f"调用通义千问API时出错: {str(e)}")
            raise Exception(f"调用通义千问API时出错: {str(e)}")

    def type(self) -> str:
        return "qwen"

def analyze_data(df, llm, question):
    try:
        smart_df = SmartDataframe(df, config={
            "llm": llm,
            "verbose": False,
            "enable_cache": False
        })
        
        result = smart_df.chat(question)
        return result
    except Exception as e:
        logger.error(f"分析过程中出现错误: {str(e)}")
        return f"分析过程中出现错误: {str(e)}"

def calculate_manual_results(df):
    results = {
        "总记录数": len(df),
        "总金额": (df['Quantity'] * df['Unit_Price']).sum(),
        "不同供应商数量": df['Supplier'].nunique(),
        "不同物品类别数量": df['Item_Category'].nunique(),
        "平均单价": df['Unit_Price'].mean(),
        "各供应商总金额": df.groupby('Supplier').apply(lambda x: (x['Quantity'] * x['Unit_Price']).sum()).to_dict(),
        "各物品类别平均单价": df.groupby('Item_Category')['Unit_Price'].mean().to_dict(),
        "最大订单金额": (df['Quantity'] * df['Unit_Price']).max(),
        "最小订单金额": (df['Quantity'] * df['Unit_Price']).min(),
        "订单金额中位数": (df['Quantity'] * df['Unit_Price']).median()
    }
    return results

def main():
    try:
        df = pd.read_csv('Procurement KPI Analysis Dataset.csv')
        llm = QwenLLM()
        
        questions = [
            "计算数据集中总共有多少条记录？",
            "计算所有订单的总金额（Quantity * Unit_Price）是多少？",
            "统计数据集中有多少个不同的供应商（Supplier）？",
            "统计数据集中有多少个不同的物品类别（Item_Category）？",
            "计算所有订单的平均单价（Unit_Price）是多少？",
            "计算每个供应商的总采购金额（Quantity * Unit_Price）是多少？",
            "计算每个物品类别（Item_Category）的平均单价是多少？",
            "找出金额最大的订单，其金额是多少？",
            "找出金额最小的订单，其金额是多少？",
            "计算所有订单金额的中位数是多少？"
        ]
        
        # 计算手动结果
        manual_results = calculate_manual_results(df)
        
        # 创建结果文件
        with open('analysis_results.txt', 'w', encoding='utf-8') as f:
            f.write("采购KPI分析结果对比\n")
            f.write("=" * 50 + "\n\n")
            
            for i, question in enumerate(questions, 1):
                f.write(f"问题 {i}: {question}\n")
                f.write("-" * 50 + "\n")
                
                # 获取AI分析结果
                ai_result = analyze_data(df, llm, question)
                f.write(f"AI分析结果: {ai_result}\n")
                
                # 获取手动计算结果
                manual_result = manual_results[list(manual_results.keys())[i-1]]
                f.write(f"手动计算结果: {manual_result}\n")
                f.write("\n")
        
        print("分析结果已保存到 analysis_results.txt 文件中")
            
    except Exception as e:
        logger.error(f"程序执行过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 