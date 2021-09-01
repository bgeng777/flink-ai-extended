#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
import os
import pandas as pd
import time
from typing import List
from joblib import dump, load
from pyflink.datastream import StreamExecutionEnvironment

from pyflink.table.udf import udf, FunctionContext
from pyflink.table import Table, ScalarFunction, DataTypes, TableConfig, StreamTableEnvironment

EXAMPLE_COLUMNS = ['sl', 'sw', 'pl', 'pw', 'type']

def main():
    exec_env = StreamExecutionEnvironment.get_execution_environment()
    exec_env.set_parallelism(1)
    t_config = TableConfig()
    t_env = StreamTableEnvironment.create(exec_env, t_config)
    t_env.get_config().get_configuration().set_string("taskmanager.memory.task.off-heap.size", '80m')
    statement_set = t_env.create_statement_set()

    # Save model to local
    model_path = '/Users/kenken/BGcodes/flink-ai-extended/flink-ai-flow/examples/tutorial_project/workflows/tutorial_workflow/saved_model/2021_08_30_15_03_10'

    data_meta = '/Users/kenken/BGcodes/flink-ai-extended/flink-ai-flow/examples/tutorial_project/resources/iris_test.csv'
    t_env.execute_sql('''
                CREATE TABLE predict_source (
                    sl FLOAT,
                    sw FLOAT,
                    pl FLOAT,
                    pw FLOAT,
                    type FLOAT
                ) WITH (
                    'connector' = 'filesystem',
                    'path' = '{uri}',
                    'format' = 'csv',
                    'csv.ignore-parse-errors' = 'true'
                )
            '''.format(uri=data_meta))
    table = t_env.from_path('predict_source')
    clf = load(model_path)
    print(clf)
    # Define the python udf

    class Predict(ScalarFunction):
        def open(self, function_context: FunctionContext):
            print("dd")
        def eval(self, sl, sw, pl, pw):
            # print(clf)
            # records = [[sl, sw, pl, pw]]
            # df = pd.DataFrame.from_records(records, columns=['sl', 'sw', 'pl', 'pw'])
            return 1.0

    # Register the udf in flink table env, so we can call it later in SQL statement
    t_env.register_function('mypred',
                              udf(f=Predict(),
                                  input_types=[DataTypes.FLOAT(), DataTypes.FLOAT(),
                                               DataTypes.FLOAT(), DataTypes.FLOAT()],
                                  result_type=DataTypes.FLOAT()))
    table = table.select("mypred(sl,sw,pl,pw)")

    table_env = t_env
    table_env.execute_sql('''
               CREATE TABLE predict_sink (
                   prediction FLOAT 
               ) WITH (
                   'connector' = 'filesystem',
                   'path' = '{uri}',
                   'format' = 'csv',
                   'csv.ignore-parse-errors' = 'true'
               )
           '''.format(uri='/Users/kenken/BGcodes/flink-ai-extended/flink-ai-flow/'
                          'examples/tutorial_project/workflows/tutorial_workflow/result')).print()
    statement_set.add_insert("predict_sink", table)
    # statement_set.execute()
    job_client = statement_set.execute().get_job_client()
    print(str(job_client.get_job_id()))
    job_client.get_job_execution_result(user_class_loader=None).result()
    print("???")


if __name__ == '__main__':
    main()