import os

import openai
from langchain.llms import OpenAI
from rich import print


class LlmNaiveCodeExecutor:
    llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

    @classmethod
    def naive_llm_code_executor(cls, code: str, input_string: str):
        prompt: str = f"""
                Given the following function definition:
                ```
                {code}
                ```
                what is the result of
                ```
                transform("{input_string}")
                ```
                ? 
                
                Return only the result in the completion. 
                I don't want any other comments. 
                Don't say "The result is" or similar remarks.
                """
        result = cls.llm(prompt=prompt, max_tokens=2048)
        return result
