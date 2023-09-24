from pydantic import BaseModel, Field
from guardrails.validators import BugFreePython
from guardrails.datatypes import PythonCode
import guardrails as gd
import openai

from rich import print

prompt = """
Given the following high level leetcode problem description, write a Python code snippet that solves the problem.

Problem Description:
${leetcode_problem}

${gr.complete_json_suffix}"""


class BugFreePythonCode(BaseModel):
    python_code: PythonCode = Field(validators=[BugFreePython(on_fail="reask")])

    class Config:
        arbitrary_types_allowed = True


guard = gd.Guard.from_pydantic(output_class=BugFreePythonCode, prompt=prompt)
# print(guard.base_prompt)

template_data="""
01-05-2023
02-05-2023
04-05-2023
06-05-2023
07-05-2023
08-05-2023
09-05-2023
10-05-2023
"""

source_data="""
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

leetcode_problem = f"""
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

#Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

raw_llm_response, validated_response = guard(
    openai.Completion.create,
    prompt_params={"leetcode_problem": leetcode_problem},
    engine="text-davinci-003",
    max_tokens=2048,
    temperature=0,
)

# print(validated_response)

print(validated_response["python_code"])

try:
    exec(validated_response["python_code"])
    print("Success!")
except Exception as e:
    print("Failed!")
