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
from typing import List

import ai_flow as af
import numpy as np
import tensorflow as tf
from ai_flow import FunctionContext
from flink_ai_flow.pyflink import SourceExecutor, FlinkFunctionContext, SinkExecutor, Executor
from pyflink.table import Table, TableEnvironment, ScalarFunction, \
    DataTypes
from pyflink.table.udf import udf
from flink_ai_flow.pyflink import TableEnvCreator, SourceExecutor, FlinkFunctionContext, Executor, \
    ExecutionEnvironment, BatchTableEnvironment
from flink_ml_tensorflow.tensorflow_TFConfig import TFConfig
from flink_ml_tensorflow.tensorflow_on_flink_mlconf import MLCONSTANTS
from flink_ml_tensorflow.tensorflow_on_flink_table import train
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings, Table, TableEnvironment


class StreamTableEnvCreator(TableEnvCreator):

    def create_table_env(self):
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        stream_env.set_parallelism(1)
        t_env = StreamTableEnvironment.create(
            stream_env,
            environment_settings=EnvironmentSettings.new_instance().in_streaming_mode().use_blink_planner().build())
        statement_set = t_env.create_statement_set()
        t_env.get_config().set_python_executable('python')
        t_env.get_config().get_configuration().set_boolean('python.fn-execution.memory.managed', True)
        return stream_env, t_env, statement_set


class StreamPreprocessSource(SourceExecutor):

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        table_env: TableEnvironment = function_context.get_table_env()
        table_env.execute_sql('''
            create table stream_train_preprocess_source (
                age varchar,
                workclass varchar,
                fnlwgt varchar,
                education varchar,
                education_num varchar,
                marital_status varchar,
                occupation varchar,
                relationship varchar,
                race varchar,
                gender varchar,
                capital_gain varchar,
                capital_loss varchar,
                hours_per_week varchar,
                native_country varchar,
                income_bracket varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'census_input_preprocess_topic',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'stream_train_preprocess_source',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        ''')
        table = table_env.from_path('stream_train_preprocess_source')
        return table


class StreamPreprocessExecutor(SinkExecutor):
    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        table_env: TableEnvironment = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        table_env.execute_sql('''
            create table stream_train_preprocess_sink (
                age varchar,
                workclass varchar,
                fnlwgt varchar,
                education varchar,
                education_num varchar,
                marital_status varchar,
                occupation varchar,
                relationship varchar,
                race varchar,
                gender varchar,
                capital_gain varchar,
                capital_loss varchar,
                hours_per_week varchar,
                native_country varchar,
                income_bracket varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'census_input_topic',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'stream_train_preprocess_sink',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        ''')
        statement_set.add_insert('stream_train_preprocess_sink', input_table)


class StreamTrainSource(SourceExecutor):

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        table_env: TableEnvironment = function_context.get_table_env()
        table_env.execute_sql('''
            create table stream_train_source (
                age varchar,
                workclass varchar,
                fnlwgt varchar,
                education varchar,
                education_num varchar,
                marital_status varchar,
                occupation varchar,
                relationship varchar,
                race varchar,
                gender varchar,
                capital_gain varchar,
                capital_loss varchar,
                hours_per_week varchar,
                native_country varchar,
                income_bracket varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'census_input_topic',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'stream_train_source',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        ''')
        table = table_env.from_path('stream_train_source')
        return table


class StreamTrainExecutor(Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        work_num = 2
        ps_num = 1
        python_file = 'census_distribute.py'
        func = 'stream_map_func'
        prop = {MLCONSTANTS.PYTHON_VERSION: '',
                MLCONSTANTS.ENCODING_CLASS: 'com.alibaba.flink.ml.operator.coding.RowCSVCoding',
                MLCONSTANTS.DECODING_CLASS: 'com.alibaba.flink.ml.operator.coding.RowCSVCoding',
                'sys:csv_encode_types': 'STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING',
                MLCONSTANTS.CONFIG_STORAGE_TYPE: MLCONSTANTS.STORAGE_ZOOKEEPER,
                MLCONSTANTS.CONFIG_ZOOKEEPER_CONNECT_STR: 'localhost:2181',
                MLCONSTANTS.CONFIG_ZOOKEEPER_BASE_PATH: '/demo',
                MLCONSTANTS.REMOTE_CODE_ZIP_FILE: "hdfs://localhost:9000/demo/code.zip"}
        env_path = None

        input_tb = function_context.t_env.from_path('stream_train_source')
        output_schema = None

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        train(function_context.get_exec_env(), function_context.get_table_env(), function_context.get_statement_set(),
              input_tb, tf_config, output_schema)


class StreamValidateExecutor(Executor):
    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        return []


class StreamPushExecutor(Executor):
    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        return []


class StreamPredictSource(SourceExecutor):

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        table_env: TableEnvironment = function_context.get_table_env()
        table_env.execute_sql('''
            create table stream_predict_source (
                age varchar,
                workclass varchar,
                fnlwgt varchar,
                education varchar,
                education_num varchar,
                marital_status varchar,
                occupation varchar,
                relationship varchar,
                race varchar,
                gender varchar,
                capital_gain varchar,
                capital_loss varchar,
                hours_per_week varchar,
                native_country varchar,
                income_bracket varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'census_input_topic',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'stream_predict_source',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        ''')
        table = table_env.from_path('stream_predict_source')
        print("##### StreamPredictSource")
        return table


class Predict(ScalarFunction):

    def __init__(self, model_path):
        super().__init__()
        self._predictor = None
        self._exported_model = None
        self._model_path = model_path

    def open(self, function_context: FunctionContext):
        self._exported_model = self._model_path.split('|')[1]
        with tf.Session() as session:
            tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], self._exported_model)
            self._predictor = tf.contrib.predictor.from_saved_model(self._exported_model)

    def eval(self, age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship,
             race, gender, capital_gain, capital_loss, hours_per_week, native_country):
        try:
            feature_dict = {
                'age': self._float_feature(value=int(age)),
                'workclass': self._bytes_feature(value=workclass.encode()),
                'fnlwgt': self._float_feature(value=int(fnlwgt)),
                'education': self._bytes_feature(value=education.encode()),
                'education_num': self._float_feature(value=int(education_num)),
                'marital_status': self._bytes_feature(value=marital_status.encode()),
                'occupation': self._bytes_feature(value=occupation.encode()),
                'relationship': self._bytes_feature(value=relationship.encode()),
                'race': self._bytes_feature(value=race.encode()),
                'gender': self._bytes_feature(value=gender.encode()),
                'capital_gain': self._float_feature(value=int(capital_gain)),
                'capital_loss': self._float_feature(value=int(capital_loss)),
                'hours_per_week': self._float_feature(value=float(hours_per_week)),
                'native_country': self._bytes_feature(value=native_country.encode()),
            }
            model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            model_input = model_input.SerializeToString()
            output_dict = self._predictor({'inputs': [model_input]})
            print(str(np.argmax(output_dict['scores'])))
            return str(np.argmax(output_dict['scores']))
        except Exception:
            return 'tf fail'

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class StreamPredictExecutor(Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        model_version = af.get_deployed_model_version("wide_and_deep")
        function_context.t_env.register_function('predict',
                                                 udf(f=Predict(model_version.model_path), input_types=[DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING()],
                                                     result_type=DataTypes.STRING()))

        return [input_list[0].select(
            'age, workclass, fnlwgt, education, education_num, marital_status, occupation, '
            'relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country, '
            'predict(age, workclass, fnlwgt, education, education_num, marital_status, occupation, '
            'relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country) as income_bracket')]


class StreamPredictSink(SinkExecutor):

    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        table_env: TableEnvironment = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        table_env.execute_sql('''
            create table stream_predict_sink (
                age varchar,
                workclass varchar,
                fnlwgt varchar,
                education varchar,
                education_num varchar,
                marital_status varchar,
                occupation varchar,
                relationship varchar,
                race varchar,
                gender varchar,
                capital_gain varchar,
                capital_loss varchar,
                hours_per_week varchar,
                native_country varchar,
                income_bracket varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'census_output_topic',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'stream_predict_sink',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        ''')
        statement_set.add_insert('stream_predict_sink', input_table)
