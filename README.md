# llmTableSchemaMapping

Table Schema mapping and data transformation using LLM

Naive pipeline to conduct the task is:
- get some (probably not the best one) alignment between columns of the template table and source table
- investigate the data format of source table column and create a function (using LLM) to transform the data to the format of template table column

### Column(s) alignment

- we don't make assumptions on the number of columns, we find some good alignment
- for each column in the template table, we find the best alignment in the source table
- local BERT model is used however we can freely use OpenAI embeddings or any other model/provider

### Limitations

- the data transformation is only done for numeric and date-like columns

### Corner cases and possible problems

- performance may heavily depend on the amount of sampled data 
- embedding-based alignment is using a subsample of the data (50 rows currently).
    We can be just unlucky to select a non-representative subsample and get a bad alignment.
- Getting the centroid of the embeddings is relatively simple method however some more involved clustering methods can be used.
- The data modeling of the source table may be not good enough and we fail to get any relevant embedding clusters.
- Given multiple data-like columns or name-like columns, we may completely fail to get proper alignment.
    To do better here, we need to consider larger subsample of the data and use some statistics as "features".
    It could be the moments of the lengths of the strings (mean, variance), the number of unique values, etc.