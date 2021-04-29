import os
import ai_flow as af

from ai_flow import ExampleSupportType, PythonObjectExecutor, ModelType
from ai_flow.common.scheduler_type import SchedulerType
from batch_train_stream_predict_executor import ExampleReader, ExampleTransformer, ModelTrainer, EvaluateExampleReader, \
    EvaluateTransformer, ModelEvaluator, ValidateExampleReader, ValidateTransformer, ModelValidator, ModelPusher, \
    PredictExampleReader, PredictTransformer, ModelPredictor, ExampleWriter
from ai_flow.common.path_util import get_file_dir
from ai_flow.model_center.entity.model_version_stage import ModelVersionEventType

EXAMPLE_URI = os.path.abspath('../..') + '/example_data/mnist_{}.npz'


def run_project(project_root_path):

    af.set_project_config_file(project_root_path + "/project.yaml")
    project_name = af.project_config().get_project_name()
    artifact_prefix = project_name + "."

    evaluate_trigger = af.external_trigger(name='evaluate')
    validate_trigger = af.external_trigger(name='validate')

    with af.global_config_file(project_root_path + '/resources/workflow_config.yaml'):
        # the config of train job is a periodic job which means it will run every 60 seconds
        with af.config('train_job'):
            # Register metadata raw training data(example) and read example(i.e. training dataset)
            train_example = af.register_example(name=artifact_prefix + 'train_example',
                                                support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                batch_uri=EXAMPLE_URI.format('train'))
            train_read_example = af.read_example(example_info=train_example,
                                                 executor=PythonObjectExecutor(python_object=ExampleReader()))

            # Transform(preprocessing) example
            train_transform = af.transform(input_data_list=[train_read_example],
                                           executor=PythonObjectExecutor(python_object=ExampleTransformer()))

            # Register model metadata and train model
            train_model = af.register_model(model_name=artifact_prefix + 'logistic-regression',
                                            model_type=ModelType.SAVED_MODEL,
                                            model_desc='logistic regression model')
            train_channel = af.train(input_data_list=[train_transform],
                                     executor=PythonObjectExecutor(python_object=ModelTrainer()),
                                     model_info=train_model)

        with af.config('eval_job'):
            # Evaluation of model
            evaluate_example = af.register_example(name=artifact_prefix + 'evaluate_example',
                                                   support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                   batch_uri=EXAMPLE_URI.format('evaluate'))
            evaluate_read_example = af.read_example(example_info=evaluate_example,
                                                    executor=PythonObjectExecutor(
                                                        python_object=EvaluateExampleReader()))
            evaluate_transform = af.transform(input_data_list=[evaluate_read_example],
                                              executor=PythonObjectExecutor(python_object=EvaluateTransformer()))
            # Register disk path used to save evaluate result
            evaluate_artifact_name = artifact_prefix + 'evaluate_artifact'
            evaluate_artifact = af.register_artifact(name=evaluate_artifact_name,
                                                     batch_uri=get_file_dir(__file__) + '/evaluate_result')
            # Evaluate model
            evaluate_channel = af.evaluate(input_data_list=[evaluate_transform],
                                           model_info=train_model,
                                           executor=PythonObjectExecutor(
                                               python_object=ModelEvaluator(evaluate_artifact_name)))

        with af.config('validate_job'):
            # Validation of model
            # Read validation dataset and validate model before it is used to predict

            validate_example = af.register_example(name=artifact_prefix + 'validate_example',
                                                   support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                   batch_uri=EXAMPLE_URI.format('evaluate'))
            validate_read_example = af.read_example(example_info=validate_example,
                                                    executor=PythonObjectExecutor(
                                                        python_object=ValidateExampleReader()))
            validate_transform = af.transform(input_data_list=[validate_read_example],
                                              executor=PythonObjectExecutor(python_object=ValidateTransformer()))
            validate_artifact_name = artifact_prefix + 'validate_artifact'
            validate_artifact = af.register_artifact(name=validate_artifact_name,
                                                     batch_uri=get_file_dir(__file__) + '/validate_model')
            validate_channel = af.model_validate(input_data_list=[validate_transform],
                                                 model_info=train_model,
                                                 executor=PythonObjectExecutor(
                                                     python_object=ModelValidator(validate_artifact_name)))
        with af.config('push_job'):
            # Push model to serving
            # Register metadata of pushed model
            push_model_artifact_name = artifact_prefix + 'push_model_artifact'
            push_model_artifact = af.register_artifact(name=push_model_artifact_name,
                                                       batch_uri=get_file_dir(__file__) + '/pushed_model')
            push_model_channel = af.push_model(model_info=train_model,
                                               executor=PythonObjectExecutor(
                                                   python_object=ModelPusher(push_model_artifact_name)))

        with af.config('predict_job'):
            # Prediction(Inference)
            predict_example = af.register_example(name=artifact_prefix + 'predict_example',
                                                  support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                  batch_uri=EXAMPLE_URI.format('predict'))
            predict_read_example = af.read_example(example_info=predict_example,
                                                   executor=PythonObjectExecutor(python_object=PredictExampleReader()))
            predict_transform = af.transform(input_data_list=[predict_read_example],
                                             executor=PythonObjectExecutor(python_object=PredictTransformer()))
            predict_channel = af.predict(input_data_list=[predict_transform],
                                         model_info=train_model,
                                         executor=PythonObjectExecutor(python_object=ModelPredictor()))
            # Save prediction result
            write_example = af.register_example(name=artifact_prefix + 'write_example',
                                                support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                batch_uri=get_file_dir(__file__) + '/predict_model')
            af.write_example(input_data=predict_channel,
                             example_info=write_example,
                             executor=PythonObjectExecutor(python_object=ExampleWriter()))

        # Define relation graph connected by control edge: train -> evaluate
        #                                                     \---> validate -> push
        #                                                     \---> predict
        # Once a round of training is done, validate and evaluate will be launched.
        # Prediction will start once the first round of training is done and
        # when pusher push a new model, the predictor will use the latest deployed model as well.
        af.model_version_control_dependency(src=validate_channel,
                                            model_version_event_type=ModelVersionEventType.MODEL_GENERATED,
                                            dependency=validate_trigger, model_name=train_model.name)
        af.model_version_control_dependency(src=evaluate_channel,
                                            model_version_event_type=ModelVersionEventType.MODEL_GENERATED,
                                            dependency=evaluate_trigger, model_name=train_model.name)

        af.stop_before_control_dependency(push_model_channel, validate_channel)
        # af.stop_before_control_dependency(predict_channel, train_channel)

    # Run workflow
    transform_dag = os.path.basename(project_root_path)

    af.deploy_to_airflow(project_root_path, dag_id=transform_dag)
    af.run(project_path=project_root_path,
           dag_id=transform_dag,
           scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    project_path = os.getcwd()
    run_project(project_path)
