from pydantic import BaseModel, Field
from guardrails.validators import BugFreePython
from guardrails.datatypes import PythonCode
import guardrails as gd
import openai

from rich import print


class BugFreePythonCode(BaseModel):
    python_code: PythonCode = Field(validators=[BugFreePython(on_fail="reask")])

    class Config:
        arbitrary_types_allowed = True


class CodeGeneratorGuardrails:
    prompt: str = """
    Given the following high level leetcode problem description, write a Python code snippet that solves the problem.

    Problem Description:
    ${leetcode_problem}

    ${gr.complete_json_suffix}"""

    guard = gd.Guard.from_pydantic(output_class=BugFreePythonCode, prompt=prompt)

    @classmethod
    def generate_transformer_code(cls, template_data: str, source_data: str) -> str:
        leetcode_problem: str = f"""
        This is a template data:
        ```
        {template_data}
        ```

        The following data needs to be transformed to the template data format:
        ```
        {source_data}
        ```

        Create a function called `transform` that takes a string as input and converts it to template data format.
        """

        try:
            raw_llm_response, validated_response = cls.guard(
                openai.Completion.create,
                prompt_params={"leetcode_problem": leetcode_problem},
                engine="text-davinci-003",
                max_tokens=2048,
                temperature=0,
            )
        except Exception as e:
            print("Error in generating code using text-davinci-003")
            return ""

        # print(validated_response["python_code"])

        try:
            exec(validated_response["python_code"])
        except Exception as e:
            print("The generated code has some errors. Please try again.")

        return validated_response["python_code"]
