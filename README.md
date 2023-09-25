# llmTableSchemaMapping

Table Schema mapping and data transformation using LLM

Current pipeline to conduct the task is:
- get some (probably not the best one) alignment between columns of the template table and source table
- assign each column one of the type category (date, text, numeric)
- investigate the data format of source table column and create a function (using LLM) to transform the data to the format of template table column
- apply the function to the source table column (using LLM, not the real code execution) 

## Usage
Install dependencies (pip is used for simplicity)
```shell
python3.X -m venv venv_llm_table
source venv_llm_table/bin/activate
pip install -r requirements.txt
```

Set OpenAI API key (it is used in the implementation)
```shell
export OPENAI_API_KEY=<KEY>
```

Go to `src` folder and run
```shell
python3 convert_table.py --source <source CSV> --template <template CSV> --target <target CSV>
```

### Column(s) alignment

- we don't make assumptions on the number of columns, we find some good alignment
- for each column in the template table, we find the best alignment in the source table
- local BERT model is used however we can freely use OpenAI embeddings or any other model/provider

### Limitations

- the data transformation is only done for numeric and date-like columns
- code "execution" is very naive and error-prone
- agents that force the LLM to produce a reasonable code in a while-true fashion are not implemented

### Corner cases and possible problems

- performance may heavily depend on the amount of sampled data 
- embedding-based alignment is using a subsample of the data (50 rows currently).
    We can be just unlucky to select a non-representative subsample and get a bad alignment.
- Getting the centroid of the embeddings is relatively simple method however some more involved clustering methods can be used.
- The data modeling of the source table may be not good enough and we fail to get any relevant embedding clusters.
- Given multiple data-like columns or name-like columns, we may completely fail to get proper alignment.
    To do better here, we need to consider larger subsample of the data and use some statistics as "features".
    It could be the moments of the lengths of the strings (mean, variance), the number of unique values, etc.
- Code executor output formatting is not reliable enough, need to go via guardrails.ai route
- The prompt for code generation can be waaaaay better

### TODO
- TESTS!!!
- proper deployment (at least containerization)
- graphana observability
- proper logging
- cost checking