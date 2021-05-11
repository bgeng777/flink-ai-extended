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
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings, Table, TableEnvironment
from census_common import get_accuracy_score, preprocess
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage
import python_ai_flow as paf
from code import census_dataset
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



class StreamValidateExecutor(paf.Executor):
    def __init__(self):
        super().__init__()
        self.path = None
        self.model_version = None
        self.model_name = None

    def setup(self, function_context: FunctionContext):
        self.model_name = function_context.node_spec.model.name
        # wide_and_deep model
        self.model_version = af.get_latest_generated_model_version(self.model_name)
        print("#### name {}".format(self.model_name))
        print("#### path {}".format(self.model_version.model_path))
        self.path = self.model_version.model_path.split('|')[1]

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        test_data = '/tmp/census_data/adult.validate'

        deployed_version = af.get_deployed_model_version(self.model_name)

        if deployed_version is not None:
            score = get_accuracy_score(self.path, test_data, 300)
            deployed_version_score = get_accuracy_score(deployed_version.model_path.split('|')[1], test_data)
            if score > deployed_version_score:
                af.update_model_version(model_name=self.model_name,
                                        model_version=self.model_version.version,
                                        current_stage=ModelVersionStage.VALIDATED)
        else:
            af.update_model_version(model_name=self.model_name,
                                    model_version=self.model_version.version,
                                    current_stage=ModelVersionStage.VALIDATED)
        print("### {}".format("stream validation done"))
        return []


class StreamPushExecutor(paf.Executor):
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        self.model_name = function_context.node_spec.model.name
        self.model_version = af.get_latest_validated_model_version(self.model_name)

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        deployed_version = af.get_deployed_model_version(self.model_name)

        if deployed_version is not None:
            af.update_model_version(model_name=self.model_name,
                                    model_version=deployed_version.version,
                                    current_stage=ModelVersionStage.DEPRECATED)

        af.update_model_version(model_name=self.model_name,
                                model_version=self.model_version.version,
                                current_stage=ModelVersionStage.DEPLOYED)

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
            arg_list = [age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship,
                        race, gender, capital_gain, capital_loss, hours_per_week, native_country]
            tmp = dict(zip(census_dataset._CSV_COLUMNS[:-1], arg_list))
            model_input = preprocess(tmp)
            output_dict = self._predictor({'inputs': [model_input]})
            print(str(np.argmax(output_dict['scores'])))
            return str(np.argmax(output_dict['scores']))
        except Exception:
            return 'tf fail'


class StreamPredictExecutor(Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        model_version = af.get_deployed_model_version('wide_and_deep')
        print("##### StreamPredictExecutor {}".format(model_version.version))
        function_context.t_env.register_function('predict',
                                                 udf(f=Predict(model_version.model_path), input_types=[DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING()],
                                                     result_type=DataTypes.STRING()))
        print("#### {}".format(self.__class__.__name__))
        return [input_list[0].select(
            'age, workclass, fnlwgt, education, education_num, marital_status, occupation, '
            'relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country, '
            'predict(age, workclass, fnlwgt, education, education_num, marital_status, occupation, '
            'relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country) as income_bracket')]
            # 'income_bracket')]


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
        print("#### {}".format(self.__class__.__name__))
