from ai_flow import ExecutionMode, ExampleSupportType, BaseJobConfig, PythonObjectExecutor, ModelType, \
    ArtifactMeta
from ai_flow.common.scheduler_type import SchedulerType

from ai_flow.model_center.entity.model_version_stage import ModelVersionEventType
from flink_ai_flow import FlinkPythonExecutor
from flink_ai_flow import LocalFlinkJobConfig
from stream_train_stream_predict_pyflink_executor import *


EXAMPLE_URI = os.path.abspath('.') + '/example_data/iris_{}.csv'


def run_pyflink_project(project_root_path):
    af.set_project_config_file(example_util.get_project_config_file(project_root_path))
    
    python_job_config = BaseJobConfig()
    python_job_config.platform = 'local'
    python_job_config.engine = 'python'
    python_job_config.exec_mode = ExecutionMode.STREAM
    
    evaluate_trigger = af.external_trigger('evaluate_trigger')
    validate_trigger = af.external_trigger('validate_trigger')
    predict_trigger = af.external_trigger("predict_trigger")

    project_name = example_util.get_parent_dir_name(__file__)
    project_meta = af.register_project(name=project_name,
                                       uri=project_root_path,
                                       project_type='local python')
    af.get_
    with af.config(python_job_config):
        train_example = af.register_example(name='train_example',
                                            support_type=ExampleSupportType.EXAMPLE_STREAM,
                                            stream_uri=EXAMPLE_URI.format('train'),
                                            data_format='csv')
        train_model = af.register_model(model_name='iris_model',
                                        model_type=ModelType.SAVED_MODEL,
                                        model_desc='hello, iris model')
        train_read_example_channel = af.read_example(example_info=train_example,
                                                     executor=PythonObjectExecutor(
                                                         python_object=ReadTrainExample.LoadTrainExample()))
        train_channel = af.train(input_data_list=[train_read_example_channel],
                                 executor=PythonObjectExecutor(python_object=TrainModel()),
                                 model_info=train_model)

        evaluate_example = af.register_example(name='evaluate_example', support_type=ExampleSupportType.EXAMPLE_STREAM,
                                               data_format='csv',
                                               batch_uri=EXAMPLE_URI.format('test'),
                                               stream_uri=EXAMPLE_URI.format('test'))
        evaluate_example_channel = af.read_example(example_info=evaluate_example, executor=PythonObjectExecutor(python_object=TestExampleReader()))
        evaluate_result = get_file_dir(__file__) + '/evaluate_result'
        if os.path.exists(evaluate_result):
            os.remove(evaluate_result)
        evaluate_artifact: ArtifactMeta = af.register_artifact(name='evaluate_artifact',
                                                               batch_uri=evaluate_result,
                                                               stream_uri=evaluate_result)
        evaluate_channel = af.evaluate(input_data_list=[evaluate_example_channel], model_info=train_model,
                                       executor=PythonObjectExecutor(python_object=EvaluateModel()))
        validate_example = af.register_example(name='validate_example', support_type=ExampleSupportType.EXAMPLE_STREAM,
                                               data_format='csv',
                                               batch_uri=EXAMPLE_URI.format('test'),
                                               stream_uri=EXAMPLE_URI.format('test'))
        validate_example_channel = af.read_example(example_info=validate_example,
                                                   executor=PythonObjectExecutor(python_object=TestExampleReader()))
        validate_result = get_file_dir(__file__) + '/validate_result'
        if os.path.exists(validate_result):
            os.remove(validate_result)
        validate_artifact: ArtifactMeta = af.register_artifact(name='validate_artifact',
                                                               batch_uri=validate_result,
                                                               stream_uri=validate_result)
        validate_channel = af.model_validate(input_data_list=[validate_example_channel], model_info=train_model,
                                             executor=PythonObjectExecutor(python_object=ValidateModel()))
    """configure local mode of job config"""
    pyflink_job_config = LocalFlinkJobConfig()
    pyflink_job_config.local_mode = 'python'
    """configure table environment create function of job config"""
    pyflink_job_config.set_table_env_create_func(StreamTableEnvCreator())

    with af.config(pyflink_job_config):
        predict_example = af.register_example(name='predict_example',
                                              support_type=ExampleSupportType.EXAMPLE_BATCH,
                                              batch_uri=EXAMPLE_URI.format('test'),
                                              stream_uri=EXAMPLE_URI.format('test'),
                                              data_format='csv')
        predict_read_example = af.read_example(example_info=predict_example,
                                               executor=FlinkPythonExecutor(python_object=Source()))
        predict_channel = af.predict(input_data_list=[predict_read_example],
                                     model_info=train_model,
                                     executor=FlinkPythonExecutor(python_object=Transformer()))

        write_example = af.register_example(name='write_example',
                                            support_type=ExampleSupportType.EXAMPLE_BATCH,
                                            batch_uri=get_file_dir(
                                                __file__) + '/predict_model.csv',
                                            stream_uri=get_file_dir(
                                                __file__) + '/predict_model.csv',
                                            data_format='csv')
        af.write_example(input_data=predict_channel,
                         example_info=write_example,
                         executor=FlinkPythonExecutor(python_object=Sink()))

    af.model_version_control_dependency(src=evaluate_channel,
                                        model_version_event_type=ModelVersionEventType.MODEL_GENERATED,
                                        dependency=evaluate_trigger, model_name=train_model.name)

    af.model_version_control_dependency(src=validate_channel,
                                        model_version_event_type=ModelVersionEventType.MODEL_GENERATED,
                                        dependency=validate_trigger, model_name=train_model.name)

    af.model_version_control_dependency(src=predict_channel,
                                        model_version_event_type=ModelVersionEventType.MODEL_DEPLOYED,
                                        dependency=predict_trigger, model_name=train_model.name)
    # Run workflow
    project_dag = project_name
    af.deploy_to_airflow(project_root_path, dag_id=project_dag)
    context = af.run(project_path=project_root_path,
                     dag_id=project_dag,
                     scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    project_path = example_util.init_project_path(".")
    run_pyflink_project(project_path)
