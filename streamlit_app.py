import streamlit as st
import pandas as pd
from io import BytesIO
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
import json

class Sentence(BaseModel):
    """Information about a generated sentence."""
    english: str = Field(description="The generated English sentence")
    chinese: str = Field(description="The Chinese translation of the English sentence")

class Data(BaseModel):
    """Extracted data about sentences."""
    sentences: List[Sentence]

def generate_sentences(api_key, text, num_sentences):
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(model="gpt-4o")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位高中英语老师。将下面的文本转换成学生可以练习的英语句子，并生成对应的中文翻译。"),
        ("user", "文本: {text}"),
        ("user", f"请生成 {num_sentences} 个句子，并以如下JSON格式返回: {{\"sentences\":[{{\"english\":\"<英语句子>\",\"chinese\":\"<中文翻译>\"}}]}}")
    ])

    output_parser = PydanticOutputParser(pydantic_object=Data)
    chain = prompt | llm | output_parser

    try:
        variables = {"text": text, "num_sentences": num_sentences}
        generated_data = chain.invoke(variables)
    except Exception as e:
        st.error(f"Error during invocation: {e}")
        return []

    st.json(generated_data.dict())  # Show the raw generated data for debugging

    sentences_data = []
    for sentence in generated_data.sentences:
        sentences_data.append({
            'english': sentence.english,
            'chinese': sentence.chinese,
            'recording': '',
            'role': '',
            'topic': ''
        })

    return sentences_data

st.title('英语句子生成工具')

api_key = st.text_input('请输入OpenAI API密钥', type='password')
text = st.text_area('请输入文本')
role = st.text_input('请输入角色（默认是0）', value='0')
topic = st.text_input('请输入主题')
num_sentences = st.number_input('请输入生成句子的数量', min_value=1, value=5)

if st.button('生成句子'):
    if not api_key or not text or not topic:
        st.error('请填写所有字段')
    else:
        sentences_data = generate_sentences(api_key, text, num_sentences)

        # 添加角色和主题到每条记录
        for sentence in sentences_data:
            sentence['role'] = role
            sentence['topic'] = topic

        if sentences_data:
            df = pd.DataFrame(sentences_data)
            st.write(df)

            # 将DataFrame导出为Excel文件
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sentences')
            output.seek(0)

            st.download_button(
                label="下载句子表格",
                data=output,
                file_name='generated_sentences.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.write('没有生成句子')
