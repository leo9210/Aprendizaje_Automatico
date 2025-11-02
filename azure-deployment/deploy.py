import os

from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.conda_dependencies import CondaDependencies


ws = Workspace.from_config()
env_file = os.path.join(os.path.dirname(__file__), "conda_env.yaml")

env = Environment.from_conda_specification(name="pycaret-env", file_path=env_file)

model = Model.register(
    workspace=ws,
    model_path="final_model.pkl",
    model_name="houses-price-predictor-model"
)

inference_config = InferenceConfig(entry_script="score.py", environment=env)

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=4)

service = Model.deploy(
    workspace=ws,
    name="houses-price-predictor-service-3",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)
