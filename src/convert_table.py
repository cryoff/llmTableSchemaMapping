from pandas import DataFrame

from aligner.embedding_aligner import EmbeddingBasedAligner
from categorizer.data_categorizer import DataCategorizer
from const.constants import DATE_CATEGORY
from transformer.code_generating_transformer import CodeGeneratingDataTransformer
from utils.data_preprocessor import Preprocessor


class TableConverter:
    def __init__(self,
                 template_csv_path: str,
                 source_csv_path: str,
                 dest_csv_path: str) -> None:
        super().__init__()
        self.template_df, self.source_df = Preprocessor.load_data(template_csv_path, source_csv_path)
        # self.aligner: EmbeddingBasedAligner = EmbeddingBasedAligner(
        #     template_df=self.template_df,
        #     source_df=self.source_df
        # )
        self.categorizer: DataCategorizer = DataCategorizer()
        self.dest_csv_path: str = dest_csv_path

    def convert(self):
        # alignment_mapping: dict[str, str] = self.aligner.get_alignment()
        # print(alignment_mapping)

        # FIXME
        # FIXME
        # FIXME
        # FIXME
        # FIXME
        # FIXME
        # FIXME
        # FIXME

        alignment_mapping: dict[str, str] = {'Date': 'PolicyDate', 'EmployeeName': 'Employee_Name', 'Plan': 'Plan_Name', 'PolicyNumber': 'PremiumAmount', 'Premium': 'Cost'}
        dest_df: DataFrame = self.source_df[alignment_mapping.values()]
        # print(dest_df.columns)

        template_data_types: dict[str, str] = self.categorizer.get_data_categories(self.template_df)
        print(template_data_types)
        dest_data_types: dict[str, str] = self.categorizer.get_data_categories(dest_df)
        print(dest_data_types)

        for col_name in template_data_types.keys():
            if template_data_types[col_name] != dest_data_types[alignment_mapping[col_name]]:
                raise ValueError("Unfortunately, the data types of the columns do not match")

        # align data format for date columns
        for col_name in dest_data_types.keys():
            if dest_data_types[col_name] == DATE_CATEGORY:
                print(f"{col_name} needs alignment")

                # CodeGeneratingDataTransformer.generate_transformer()


if __name__ == "__main__":
    template_csv_path: str = "/home/users/andreya1/Documents/PERSONAL/llmTestTask/llmTableSchemaMapping/task/template.csv"
    # source_csv_path: str = "/home/users/andreya1/Documents/PERSONAL/llmTestTask/llmTableSchemaMapping/task/table_A.csv"
    source_csv_path: str = "/home/users/andreya1/Documents/PERSONAL/llmTestTask/llmTableSchemaMapping/task/table_B.csv"
    dest_pdf_path: str = "path_to_dest_pdf"
    # naive_iteration: EmbeddingBasedAligner = EmbeddingBasedAligner(template_csv_path, source_csv_path, dest_pdf_path)
    # alignment_mapping: dict[str, str] = naive_iteration.get_alignment()
    # print(alignment_mapping)
    converter: TableConverter = TableConverter(template_csv_path, source_csv_path, dest_pdf_path)
    converter.convert()
