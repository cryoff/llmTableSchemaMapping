import os

import openai
from langchain.llms import OpenAI
from rich import print


class ChatGptTransformationCodeGenerator:
    llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

    @classmethod
    def generate_transformer_code(cls, template_data: str, source_data: str) -> str:

        leetcode_problem = f"""
        Consider the following examples of the template data (samples are divided by line break):
        ```
        {template_data}
        ```
        The task is to convert the input string to the format of the template data.

        The following samples (line break separated) are examples of the possible input:
        ```
        {source_data}
        ```

        Create a function called `transform` that takes a string as input and converts it to template data format.
        If it is the same, let a function return the input string.
        Please write generic function not a function that works only for the given input string.
        Input string is always a single string, not a list of strings.
        Input string does not contain any line breaks. 
        Function code shall not contain any kind of splitting by line break instructions.
        Ensure that the result string is in the same format as the template string.
        Ensure that the result string does not contain any line breaks or special characters.
        Ensure that the result string is a single string, not a list of strings.

        The function definition should be as follows:
        ```
        def transform(input_string: str) -> str:
        ```

        Return only the code in the completion. I don't want any other comments. Don't say "here is your code" or similar remarks.
        """
        code = cls.llm(prompt=leetcode_problem, max_tokens=2048)

        try:
            cls.validate_code(code, source_data.split("\n")[0])
        except Exception as e:
            print("Code generated using gpt-3.5-turbo has errors")
            return ""

        return code

    @classmethod
    def validate_code(cls, code: str, sample: str) -> str:
        try:
            exec(code, {'input_string': sample})
        except Exception as e:
            print("The generated code has some errors. Please try again.")
            return ""

        return code


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    chatgpt_transformer = ChatGptTransformationCodeGenerator()
    template_data = """
    01-05-2023
    02-05-2023
    04-05-2023
    06-05-2023
    07-05-2023
    08-05-2023
    09-05-2023
    10-05-2023
    """

    source_data = """
    2023-05-01
    2023-05-02
    2023-05-03
    2023-05-04
    2023-05-05
    2023-05-06
    2023-05-07
    2023-05-08
    2023-05-09
    2023-05-10
    """

    r_code: str = chatgpt_transformer.generate_transformer_code(template_data=template_data, source_data=source_data)
    print(r_code)
    source_sample: str = source_data.split("\n")[0]
    chatgpt_transformer.validate_code(code=r_code, sample=source_sample)
