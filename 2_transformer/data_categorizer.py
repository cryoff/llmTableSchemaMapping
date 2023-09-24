import openai
from guardrails import Guard
from guardrails.validators import ValidChoices
from rich import print


class DataCategorizer:
    DATE_CATEGORY: str = "date"
    NUMBER_CATEGORY: str = "number"
    TEXT_CATEGORY: str = "text"

    @classmethod
    def get_data_category(cls, data):
        data_types: list[str] = [cls.DATE_CATEGORY, cls.NUMBER_CATEGORY, cls.TEXT_CATEGORY]

        # capitalize the first letter for each word in data_types
        data_types.extend([data_type.capitalize() for data_type in data_types])

        guard = Guard.from_string(
            validators=[ValidChoices(choices=data_types, on_fail='reask')],
            description="Yes-no question",
            prompt=f"Which data type describes the slice of data better?"
                   f"Data sample: ```{data}```"
                   f"Please find the best data type from the following list: {data_types}",
            num_reasks=10
        )

        raw_llm_output, validated_llm_response = guard(openai.Completion.create)
        print(validated_llm_response.lower())
        print(guard.state.most_recent_call.tree)


if __name__ == "__main__":
    # DataTransformer.is_same_format("test")
    # DataTransformer.is_same_format("2023-05-04")
    # data1: str = "2023-05-02"
    # data2: str = "06-05-2023"

    # data1: str = "EF10111"
    # data2: str = "KL14141"
    DataCategorizer.get_data_category("Carol Martinez")
