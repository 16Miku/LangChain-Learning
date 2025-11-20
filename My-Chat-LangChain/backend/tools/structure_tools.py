from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List
import json

# 定义 Pydantic 模型作为参数 Schema

class PaperAnalysisSchema(BaseModel):
    title: str = Field(description="论文的完整标题")
    authors: List[str] = Field(description="论文的核心作者列表")
    research_field: str = Field(description="根据内容总结出的研究方向")
    summary: str = Field(description="对论文核心贡献的详细总结")
    author_contact: str = Field(description="从抓取内容中找到的作者邮箱或个人主页，如果找不到则为 '联系方式未找到'")

class LinkedinProfileSchema(BaseModel):
    full_name: str = Field(description="用户的全名")
    headline: str = Field(description="用户的头衔或当前职位")
    location: str = Field(description="用户所在的地理位置")
    summary: str = Field(description="个人简介部分的总结")
    experience: List[str] = Field(description="一个包含所有工作经历的列表")
    contact: str = Field(description="从抓取内容中找到的邮箱或个人主页，如果找不到则为 '联系方式未找到'")

# 定义实际的可执行工具

@tool(args_schema=PaperAnalysisSchema)
def format_paper_analysis(title: str, authors: List[str], research_field: str, summary: str, author_contact: str):
    """当用户要求对一篇学术论文进行详细分析时，必须调用此工具来格式化最终报告。
    此工具不执行分析，仅用于输出结构化数据。"""
    result = {
        "type": "paper_analysis",
        "data": {
            "title": title,
            "authors": authors,
            "research_field": research_field,
            "summary": summary,
            "author_contact": author_contact
        }
    }
    # 返回 JSON 字符串以便 Agent 最终输出，或者前端可以解析
    return json.dumps(result, ensure_ascii=False)

@tool(args_schema=LinkedinProfileSchema)
def format_linkedin_profile(full_name: str, headline: str, location: str, summary: str, experience: List[str], contact: str):
    """当用户要求提取领英个人主页信息时，必须调用此工具来格式化最终报告。
    此工具不执行提取，仅用于输出结构化数据。"""
    result = {
        "type": "linkedin_profile",
        "data": {
            "full_name": full_name,
            "headline": headline,
            "location": location,
            "summary": summary,
            "experience": experience,
            "contact": contact
        }
    }
