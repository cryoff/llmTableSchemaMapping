# llmTableSchemaMapping

Table Schema mapping and data transformation using LLM

Naive pipeline to conduct the task is:
- get some (probably not the best one) alignment between columns of the template table and source table
- investigate the data format of source table column and create a function (using LLM) to transform the data to the format of template table column

### Column(s) alignment

- we don't make assumptions on the number of columns, we find some good alignment
- for each column in the template table, we find the best alignment in the source table