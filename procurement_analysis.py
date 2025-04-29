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
    """通义千问LLM实现"""
    
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
    """计算手动分析结果"""
    return {
        "各供应商总金额": df.groupby('Supplier').apply(lambda x: (x['Quantity'] * x['Unit_Price']).sum()).to_dict(),
        "各物品类别平均单价": df.groupby('Item_Category')['Unit_Price'].mean().to_dict(),
        "订单金额中位数": (df['Quantity'] * df['Unit_Price']).median()
    }

def analyze_and_save(df, llm, questions, manual_results=None, result_mapping=None, filename=None, section_title=None):
    """分析问题并保存结果到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("采购KPI分析结果对比\n")
        f.write("=" * 50 + "\n\n")
        
        if section_title:
            f.write(f"{section_title}\n")
            f.write("=" * 50 + "\n\n")
        
        for i, question in enumerate(questions, 1):
            f.write(f"问题 {i}: {question}\n")
            f.write("-" * 50 + "\n")
            
            # 获取AI分析结果
            ai_result = analyze_data(df, llm, question)
            f.write(f"AI分析结果: {ai_result}\n")
            
            # 如果有手动结果映射，添加手动计算结果
            if manual_results and result_mapping and i-1 in result_mapping:
                manual_result = manual_results[result_mapping[i-1]]
                f.write(f"手动计算结果: {manual_result}\n")
                
            f.write("\n")

def main():
    try:
        # 加载数据
        df = pd.read_csv('Procurement KPI Analysis Dataset.csv')
        llm = QwenLLM()
        
        # 定义问题
        difficult_questions = [
            "计算每个供应商的总采购金额是多少？",
            "计算每个物品类别的平均单价是多少？",
            "计算所有订单金额的中位数是多少？"
        ]
        
        trend_questions = [
            "分析各供应商的采购金额随时间的变化趋势",
            "分析各物品类别的采购数量随时间的变化趋势",
            "分析平均单价随时间的变化趋势"
        ]
        
        # 计算手动结果
        manual_results = calculate_manual_results(df)
        
        # 定义问题与手动结果的对应关系
        question_result_mapping = {
            0: "各供应商总金额",
            1: "各物品类别平均单价",
            2: "订单金额中位数"
        }
        
        # 分析最难的问题并保存结果
        analyze_and_save(
            df, llm, difficult_questions, 
            manual_results, question_result_mapping,
            'difficult_analysis_results.txt', 
            "三个最难的问题分析结果"
        )
        
        # 分析趋势问题并保存结果
        analyze_and_save(
            df, llm, trend_questions,
            filename='trend_analysis_results.txt',
            section_title="三个趋势分析问题结果"
        )
        
        # 将所有结果合并到一个文件中
        with open('all_analysis_results.txt', 'w', encoding='utf-8') as f:
            f.write("采购KPI分析结果汇总\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入最难的问题分析结果
            f.write("三个最难的问题分析结果：\n")
            f.write("=" * 50 + "\n\n")
            
            for i, question in enumerate(difficult_questions, 1):
                f.write(f"问题 {i}: {question}\n")
                f.write("-" * 50 + "\n")
                
                # 获取AI分析结果
                ai_result = analyze_data(df, llm, question)
                f.write(f"AI分析结果: {ai_result}\n")
                
                # 获取手动计算结果
                if i-1 in question_result_mapping:
                    manual_result = manual_results[question_result_mapping[i-1]]
                    f.write(f"手动计算结果: {manual_result}\n")
                    
                f.write("\n")
            
            # 写入趋势分析问题结果
            f.write("三个趋势分析问题结果：\n")
            f.write("=" * 50 + "\n\n")
            
            for i, question in enumerate(trend_questions, 1):
                f.write(f"趋势问题 {i}: {question}\n")
                f.write("-" * 50 + "\n")
                
                # 获取AI分析结果
                ai_result = analyze_data(df, llm, question)
                f.write(f"AI分析结果: {ai_result}\n")
                f.write("\n")
        
        print("分析结果已保存到文件中")
            
    except Exception as e:
        logger.error(f"程序执行过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 