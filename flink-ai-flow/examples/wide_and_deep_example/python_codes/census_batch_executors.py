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

import tensorflow as tf
import ai_flow as af
from typing import List
from python_ai_flow import FunctionContext, Executor
import python_ai_flow as paf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from flink_ai_flow.pyflink import TableEnvCreator, SourceExecutor, FlinkFunctionContext, \
    ExecutionEnvironment, BatchTableEnvironment
import flink_ai_flow.pyflink as faf
from flink_ml_tensorflow.tensorflow_TFConfig import TFConfig
from flink_ml_tensorflow.tensorflow_on_flink_mlconf import MLCONSTANTS
from flink_ml_tensorflow.tensorflow_on_flink_table import train
from ai_flow.client.ai_flow_client import get_ai_flow_client
from pyflink.table import StreamTableEnvironment, EnvironmentSettings, Table, TableEnvironment
from ai_flow.common.path_util import get_file_dir
import census_dataset
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage
from notification_service.base_notification import DEFAULT_NAMESPACE, BaseEvent


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def preprocess(input_dict):

    feature_dict = {
        'age': _float_feature(value=int(input_dict['age'])),
        'workclass': _bytes_feature(value=input_dict['workclass'].encode()),
        'fnlwgt': _float_feature(value=int(input_dict['fnlwgt'])),
        'education': _bytes_feature(value=input_dict['education'].encode()),
        'education_num': _float_feature(value=int(input_dict['education_num'])),
        'marital_status': _bytes_feature(value=input_dict['marital_status'].encode()),
        'occupation': _bytes_feature(value=input_dict['occupation'].encode()),
        'relationship': _bytes_feature(value=input_dict['relationship'].encode()),
        'race': _bytes_feature(value=input_dict['race'].encode()),
        'gender': _bytes_feature(value=input_dict['gender'].encode()),
        'capital_gain': _float_feature(value=int(input_dict['capital_gain'])),
        'capital_loss': _float_feature(value=int(input_dict['capital_loss'])),
        'hours_per_week': _float_feature(value=float(input_dict['hours_per_week'])),
        'native_country': _bytes_feature(value=input_dict['native_country'].encode()),
    }
    model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    model_input = model_input.SerializeToString()
    return model_input


def get_accuracy_score(model_path, data_path):
    predictor = tf.contrib.predictor.from_saved_model(model_path)
    data = pd.read_csv(data_path, names=census_dataset._CSV_COLUMNS)
    label = data.pop('income_bracket')
    label = label.map({'<=50K': 0, '>50K': 1})
    inputs = []
    for _, row in data.iterrows():
        tmp = dict(zip(census_dataset._CSV_COLUMNS[:-1], row))
        tmp = preprocess(tmp)
        inputs.append(tmp)
    output_dict = predictor({'inputs': inputs})
    res = [np.argmax(output_dict['scores'][i]) for i in range(0, len(output_dict['scores']))]
    return accuracy_score(label, res)


class BatchTableEnvCreator(TableEnvCreator):

    def create_table_env(self):
        batch_env = ExecutionEnvironment.get_execution_environment()
        batch_env.setParallelism(1)
        t_env = BatchTableEnvironment.create(
            batch_env,
            environment_settings=EnvironmentSettings.new_instance().in_batch_mode().use_blink_planner().build())
        statement_set = t_env.create_statement_set()
        t_env.get_config().set_python_executable('python')
        t_env.get_config().get_configuration().set_boolean('python.fn-execution.memory.managed', True)
        return batch_env, t_env, statement_set


class BatchPreprocessExecutor(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        data_path = '/tmp/census_data/adult.data'
        df = pd.read_csv(data_path, header=None)
        df = shuffle(df)
        df.to_csv('/tmp/census_data/adult.data', index=False, header=None)
        get_ai_flow_client().send_event(BaseEvent(key='wide_and_deep_base', value='BATCH_PREPROCESS',
                                                  event_type='BATCH_PREPROCESS',
                                                  namespace=DEFAULT_NAMESPACE))
        return []


class BatchTrainExecutor(faf.Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        work_num = 2
        ps_num = 1
        python_file = 'census_distribute.py'
        func = 'batch_map_func'
        prop = {MLCONSTANTS.PYTHON_VERSION: '', MLCONSTANTS.CONFIG_STORAGE_TYPE: MLCONSTANTS.STORAGE_ZOOKEEPER,
                MLCONSTANTS.CONFIG_ZOOKEEPER_CONNECT_STR: 'localhost:2181',
                MLCONSTANTS.CONFIG_ZOOKEEPER_BASE_PATH: '/demo',
                MLCONSTANTS.REMOTE_CODE_ZIP_FILE: 'hdfs://localhost:9000/demo/code.zip'}
        env_path = None

        input_tb = None
        output_schema = None

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        train(function_context.get_exec_env(), function_context.get_table_env(), function_context.get_statement_set(),
              input_tb, tf_config, output_schema)


class BatchEvaluateExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.path = None
        self.model_version = None
        self.model_name = None

    def setup(self, function_context: FunctionContext):
        self.model_name = function_context.node_spec.model.name
        self.model_version = af.get_latest_generated_model_version(self.model_name)
        print("#### name {}".format(self.model_name))
        print("#### path {}".format(self.model_version.model_path))
        self.path = self.model_version.model_path.split('|')[1]

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        test_data = '/tmp/census_data/adult.evaluate'
        score = get_accuracy_score(self.path, test_data)
        path = get_file_dir(__file__) + '/batch_evaluate_result'
        with open(path, 'a') as f:
            f.write(str(score) + '  -------->  ' + self.model_version.version)
            f.write('\n')
        af.update_model_version(model_name=self.model_name,
                                model_version=self.model_version.version,
                                current_stage=ModelVersionStage.VALIDATED)
        return []


class BatchValidateExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.path = None
        self.model_version = None
        self.model_name = None

    def setup(self, function_context: FunctionContext):
        self.model_name = function_context.node_spec.model.name
        self.model_version = af.get_latest_validated_model_version(self.model_name)
        print("#### name {}".format(self.model_name))
        print("#### path {}".format(self.model_version.model_path))
        print("#### ver {}".format(self.model_version.version))
        self.path = self.model_version.model_path.split('|')[1]

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        test_data = '/tmp/census_data/adult.validate'
        score = get_accuracy_score(self.path, test_data)

        path = get_file_dir(__file__) + '/batch_validate_result'
        with open(path, 'a') as f:
            f.write(str(score) + '  -------->  ' + self.model_version.version)
            f.write('\n')
        deployed_version = af.get_deployed_model_version(self.model_name)

        if deployed_version is not None:
            deployed_version_score = get_accuracy_score(deployed_version.model_path.split('|')[1], test_data)
            if score > deployed_version_score:
                af.update_model_version(model_name=self.model_name,
                                        model_version=deployed_version.version,
                                        current_stage=ModelVersionStage.DEPRECATED)
                af.update_model_version(model_name=self.model_name,
                                        model_version=self.model_version.version,
                                        current_stage=ModelVersionStage.DEPLOYED)
                with open(path, 'a') as f:
                    f.write('version {} pass validation.'.format(self.model_version.version))
                    f.write('\n')
            else:
                with open(path, 'a') as f:
                    f.write('version {} does not pass validation.'.format(self.model_version.version))
                    f.write('\n')
        else:
            af.update_model_version(model_name=self.model_name,
                                    model_version=self.model_version.version,
                                    current_stage=ModelVersionStage.DEPLOYED)
            with open(path, 'a') as f:
                f.write('version {} pass validation.'.format(self.model_version.version))
                f.write('\n')
        return []

