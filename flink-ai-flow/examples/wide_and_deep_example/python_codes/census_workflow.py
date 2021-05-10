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
import ai_flow as af
from ai_flow.common.scheduler_type import SchedulerType
from ai_flow import PythonObjectExecutor
from flink_ai_flow import LocalFlinkJobConfig, FlinkPythonExecutor
from ai_flow.model_center.entity.model_version_stage import ModelVersionEventType
from ai_flow.graph.edge import TaskAction, EventLife, MetValueCondition, MetCondition, DEFAULT_NAMESPACE


from census_batch_executors import BatchPreprocessExecutor, BatchTrainExecutor, BatchEvaluateExecutor, \
    BatchValidateExecutor, BatchTableEnvCreator
from census_stream_executors import StreamPreprocessExecutor, StreamValidateExecutor, StreamPushExecutor, \
    StreamTableEnvCreator, StreamTrainSource, StreamTrainExecutor, StreamPredictSource, StreamPredictExecutor, \
    StreamPredictSink


def get_project_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_workflow():
    af.set_project_config_file(get_project_path() + '/project.yaml')
    with af.global_config_file(config_path=get_project_path() + '/resources/workflow_config.yaml'):
        stream_train_input = af.get_example_by_name('stream_train_input')
        stream_predict_input = af.get_example_by_name('stream_predict_input')
        stream_predict_output = af.get_example_by_name('stream_predict_output')

        batch_model_info = af.get_model_by_name('wide_and_deep_base')

        workflow_config = af.default_af_job_context().global_workflow_config

        """Batch Job Configs"""
        batch_preprocess_config = workflow_config.job_configs['census_batch_preprocess']

        batch_train_config: LocalFlinkJobConfig = workflow_config.job_configs['census_batch_train']
        # Currently tf-on-flink only supports stream table env
        batch_train_config.set_table_env_create_func(StreamTableEnvCreator())

        batch_evaluate_config = workflow_config.job_configs['census_batch_evaluate']

        batch_validate_config = workflow_config.job_configs['census_batch_validate']

        """Batch Train Jobs"""
        with af.config(config=batch_preprocess_config):
            batch_preprocess_channel = af.user_define_operation(input_data_list=[],
                                                                executor=PythonObjectExecutor(
                                                                    python_object=BatchPreprocessExecutor()),
                                                                name='census_batch_preprocess')
        with af.config(config=batch_train_config):
            batch_train_channel = af.train(input_data_list=[],
                                           executor=FlinkPythonExecutor(python_object=BatchTrainExecutor()),
                                           model_info=batch_model_info, name='census_batch_train')

        with af.config(config=batch_evaluate_config):
            batch_evaluate_channel = af.evaluate(input_data_list=[],
                                                 executor=PythonObjectExecutor(python_object=BatchEvaluateExecutor()),
                                                 model_info=batch_model_info, name='census_batch_evaluate')
        with af.config(config=batch_validate_config):
            batch_validate_channel = af.model_validate(input_data_list=[],
                                                       executor=PythonObjectExecutor(
                                                           python_object=BatchValidateExecutor()),
                                                       model_info=batch_model_info, name='census_batch_validate')

        # """Stream Job Configs"""
        # stream_preprocess_config = workflow_config.job_configs['census_stream_preprocess_train']
        #
        # stream_train_config: LocalFlinkJobConfig = workflow_config.job_configs['census_stream_train']
        # stream_train_config.set_table_env_create_func(StreamTableEnvCreator())
        #
        # stream_validate_config: LocalFlinkJobConfig = workflow_config.job_configs['census_stream_validate']
        # stream_validate_config.set_table_env_create_func(StreamTableEnvCreator())
        #
        # stream_predict_config: LocalFlinkJobConfig = workflow_config.job_configs['census_stream_predict']
        # stream_predict_config.set_table_env_create_func(StreamTableEnvCreator())
        #
        # stream_push_config: LocalFlinkJobConfig = workflow_config.job_configs['census_stream_push']
        # stream_push_config.set_table_env_create_func(StreamTableEnvCreator())
        #
        # """Stream Train Jobs"""
        # with af.config(config=stream_preprocess_config):
        #     stream_preprocess_channel = af.user_define_operation(input_data_list=[],
        #                                                          executor=FlinkPythonExecutor(
        #                                                              python_object=StreamPreprocessExecutor()),
        #                                                          name='census_stream_preprocess')
        # with af.config(config=stream_train_config):
        #     stream_train_source = af.read_example(example_info=stream_train_input,
        #                                           executor=FlinkPythonExecutor(python_object=StreamTrainSource()))
        #     stream_train_channel = af.train(input_data_list=[stream_train_source],
        #                                     model_info=model_info,
        #                                     executor=FlinkPythonExecutor(python_object=StreamTrainExecutor()))
        # with af.config(config=stream_validate_config):
        #     stream_validate_channel = af.model_validate(input_data_list=[],
        #                                                 executor=FlinkPythonExecutor(
        #                                                     python_object=StreamValidateExecutor()),
        #                                                 model_info=model_info, name='census_batch_validate')
        # with af.config(config=stream_push_config):
        #     stream_push_channel = af.push_model(executor=FlinkPythonExecutor(python_object=StreamPushExecutor()),
        #                                         model_info=model_info, name='census_batch_validate')
        #
        # """Stream Prediction"""
        # with af.config(config=stream_predict_config):
        #     stream_predict_source = af.read_example(example_info=stream_predict_input,
        #                                             executor=FlinkPythonExecutor(python_object=StreamPredictSource()))
        #     stream_predict_channel = af.predict(input_data_list=[stream_predict_source],
        #                                         model_info=model_info,
        #                                         executor=FlinkPythonExecutor(python_object=StreamPredictExecutor()))
        #     stream_predict_sink = af.write_example(input_data=stream_predict_channel,
        #                                            example_info=stream_predict_output,
        #                                            executor=FlinkPythonExecutor(python_object=StreamPredictSink()))


        af.user_define_control_dependency(src=batch_train_channel,
                                          dependency=batch_preprocess_channel,
                                          event_key="wide_and_deep_base",
                                          event_type="BATCH_PREPROCESS",
                                          event_value="BATCH_PREPROCESS",
                                          action=TaskAction.RESTART,
                                          value_condition= MetValueCondition.UPDATE,
                                          life=EventLife.REPEATED
                                          )
        af.model_version_control_dependency(src=batch_evaluate_channel, dependency=batch_train_channel,
                                            model_name='wide_and_deep_base',
                                            model_version_event_type=ModelVersionEventType.MODEL_GENERATED)
        af.model_version_control_dependency(src=batch_validate_channel, dependency=batch_evaluate_channel,
                                            model_name='wide_and_deep_base',
                                            model_version_event_type=ModelVersionEventType.MODEL_VALIDATED)
        # af.model_version_control_dependency(src=stream_train_channel, dependency=batch_validate_channel,
        #                                     model_name='wide_and_deep_batch',
        #                                     model_version_event_type=ModelVersionEventType.MODEL_DEPLOYED)
        #
        # af.model_version_control_dependency(src=stream_validate_channel, dependency=stream_train_channel,
        #                                     model_name='wide_and_deep_stream',
        #                                     model_version_event_type=ModelVersionEventType.MODEL_GENERATED)
        # af.model_version_control_dependency(src=stream_push_channel, dependency=stream_validate_channel,
        #                                     model_name='wide_and_deep_stream',
        #                                     model_version_event_type=ModelVersionEventType.MODEL_VALIDATED)
        # af.model_version_control_dependency(src=stream_predict_sink, dependency=stream_push_channel,
        #                                     model_name='wide_and_deep_stream',
        #                                     model_version_event_type=ModelVersionEventType.MODEL_DEPLOYED)



        # af.stop_before_control_dependency(stream_train_channel, batch_train_channel)
        # af.stop_before_control_dependency(stream_predict_sink, batch_train_channel)
        # af.model_version_control_dependency(src=stream_predict_sink, dependency=stream_train_channel,
        #                                     model_name='wide_and_deep',
        #                                     model_version_event_type=ModelVersionEventType.MODEL_DEPLOYED)

        wide_deep_dag = 'census_workflow'

        af.deploy_to_airflow(get_project_path(), dag_id=wide_deep_dag)
        af.run(get_project_path(), dag_id=wide_deep_dag, scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    print(get_project_path())
    run_workflow()
