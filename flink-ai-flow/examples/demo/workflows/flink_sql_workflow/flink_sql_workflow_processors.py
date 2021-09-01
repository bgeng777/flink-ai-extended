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

import ai_flow as af
from pyflink.table.udf import udf
from pyflink.table import Table, ScalarFunction, DataTypes
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage
from ai_flow.util.path_util import get_file_dir
from ai_flow_plugins.job_plugins.flink import UDFWrapper
from ai_flow_plugins.job_plugins.python.python_processor import ExecutionContext, PythonProcessor
from ai_flow_plugins.job_plugins import flink
from sklearn.neighbors import KNeighborsClassifier

EXAMPLE_COLUMNS = ['sl', 'sw', 'pl', 'pw', 'type']
flink.set_flink_env(flink.FlinkStreamEnv())



class DatasetReader(PythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        """
        Read dataset using pandas
        """
        # Gets the registered dataset meta info
        dataset_meta: af.DatasetMeta = execution_context.config.get('dataset')
        # Read the file using pandas
        train_data = pd.read_csv(dataset_meta.uri, header=0, names=EXAMPLE_COLUMNS)
        # Prepare dataset
        y_train = train_data.pop(EXAMPLE_COLUMNS[4])
        return [[train_data.values, y_train.values]]


class ModelTrainer(PythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        """
        Train and save KNN model
        """
        model_meta: af.ModelMeta = execution_context.config.get('model_info')
        clf = KNeighborsClassifier(n_neighbors=5)
        x_train, y_train = input_list[0][0], input_list[0][1]
        clf.fit(x_train, y_train)

        # Save model to local
        model_path = get_file_dir(__file__) + '/saved_model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        model_path = model_path + '/' + model_timestamp
        dump(clf, model_path)
        af.register_model_version(model=model_meta, model_path=model_path)
        return []


class Source(flink.FlinkSqlProcessor):

    def sql_statements(self, execution_context: ExecutionContext) -> str:
        data_meta = execution_context.config['dataset']
        sql_statements = '''
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
                        '''.format(uri=data_meta.uri)
        return sql_statements


class Sink(flink.FlinkSqlProcessor):

    def udf_list(self, execution_context: ExecutionContext) -> List:
        model_name = execution_context.config['model_info'].name
        model_path = af.get_deployed_model_version(model_name).model_path
        clf = load(model_path)

        # Define the python udf
        class Predict(ScalarFunction):
            def eval(self, sl, sw, pl, pw):
                records = [[sl, sw, pl, pw]]
                df = pd.DataFrame.from_records(records, columns=['sl', 'sw', 'pl', 'pw'])
                return clf.predict(df)[0]

        udf_func = UDFWrapper('mypred', udf(f=Predict(),
                                            input_types=[DataTypes.FLOAT(), DataTypes.FLOAT(),
                                                         DataTypes.FLOAT(), DataTypes.FLOAT()],
                                            result_type=DataTypes.FLOAT()))
        return [udf_func]

    def sql_statements(self, execution_context: ExecutionContext) -> str:
        sql_statements = '''
                   CREATE TABLE predict_sink (
                       prediction FLOAT 
                   ) WITH (
                       'connector' = 'filesystem',
                       'path' = '{uri}',
                       'format' = 'csv',
                       'csv.ignore-parse-errors' = 'true'
                   );
                   INSERT INTO predict_sink
                    SELECT mypred(sl,sw,pl,pw)
                    FROM predict_source;
                    '''.format(uri=execution_context.config['dataset'].uri)
        return sql_statements
