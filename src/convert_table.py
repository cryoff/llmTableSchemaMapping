import argparse
from pandas import DataFrame

from aligner.embedding_aligner import EmbeddingBasedAligner
from categorizer.data_categorizer import DataCategorizer
from const.constants import DATE_CATEGORY, NUMBER_CATEGORY
from transformer.code_executor import LlmNaiveCodeExecutor
from transformer.code_generator_chatgpt import ChatGptTransformationCodeGenerator
from utils.data_preprocessor import Preprocessor


class TableConverter:
    def __init__(self,
                 template_csv_path: str,
                 source_csv_path: str,
                 dest_csv_path: str) -> None:
        super().__init__()
        self.template_df, self.source_df = Preprocessor.load_data(template_csv_path, source_csv_path)
        self.aligner: EmbeddingBasedAligner = EmbeddingBasedAligner(
            template_df=self.template_df,
            source_df=self.source_df
        )
        self.categorizer: DataCategorizer = DataCategorizer()
        self.dest_csv_path: str = dest_csv_path

    @classmethod
    def execute_code(cls, code_string: str):
        result_variables = {}
        exec(code_string, globals(), result_variables)
        return result_variables

    @classmethod
    def _apply_converter(cls, converter_code: str, input_string: str) -> str:
        res: str = LlmNaiveCodeExecutor.naive_llm_code_executor(code=converter_code, input_string=input_string)
        res.replace("\"", "")
        res.replace("\'", "")
        return res

    def convert_data(self,
                     alignment_mapping: dict[str, str],
                     template_data_types: dict[str, str],
                     dest_data_types: dict[str, str],
                     dest_df: DataFrame):

        for col_name in template_data_types.keys():
            dest_col: str = alignment_mapping[col_name]
            if dest_data_types[dest_col] == DATE_CATEGORY \
                    or dest_data_types[dest_col] == NUMBER_CATEGORY:
                print(f"{dest_col} may need conversion")

                template_sample: str = self.template_df[col_name].to_string(index=False)
                source_sample: str = self.source_df[alignment_mapping[col_name]].to_string(index=False)

                converter_code = ChatGptTransformationCodeGenerator.generate_transformer_code(
                    template_data=template_sample,
                    source_data=source_sample)
                if converter_code:
                    dest_df[dest_col] = self.source_df[dest_col].apply(
                        lambda x: self._apply_converter(converter_code, x)
                    )
                else:
                    raise ValueError("Unfortunately, the generated code has some errors. Please try again.")

    def convert(self):
        print("Search for the column alignment")
        alignment_mapping: dict[str, str] = self.aligner.get_alignment()
        print(f"Columns alignment: {alignment_mapping}")
        reverse_alignment_mapping: dict[str, str] = {v: k for k, v in alignment_mapping.items()}

        template_data_types: dict[str, str] = self.categorizer.get_data_categories(self.template_df)
        print(f"Generic data types of template: {template_data_types}")
        dest_df: DataFrame = self.source_df[alignment_mapping.values()]
        dest_data_types: dict[str, str] = self.categorizer.get_data_categories(dest_df)
        # print(dest_data_types)

        for col_name in template_data_types.keys():
            if template_data_types[col_name] != dest_data_types[alignment_mapping[col_name]]:
                raise ValueError("Unfortunately, the data types of the columns do not match")

        print("Convert data to template format")
        self.convert_data(alignment_mapping, template_data_types, dest_data_types, dest_df)
        dest_df.rename(columns=reverse_alignment_mapping, inplace=True)
        # print(dest_df)
        dest_df.to_csv(self.dest_csv_path, index=False)


def main(template_csv_path: str, source_csv_path: str, dest_pdf_path: str):
    converter = TableConverter(template_csv_path, source_csv_path, dest_pdf_path)
    converter.convert()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to PDF")

    parser.add_argument("--template", type=str, required=True, help="Path to the template CSV file")
    parser.add_argument("--source", type=str, required=True, help="Path to the source CSV file")
    parser.add_argument("--target", type=str, required=True, help="Path to the destination PDF file")

    args = parser.parse_args()

    main(args.template, args.source, args.target)
