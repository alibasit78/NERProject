from ner.components.model_pusher import ModelPusher
from ner.entity.artifact_entity import (
    ModelEvalArtifact,
    ModelTrainingArtifacts,
)
from ner.entity.config_entity import (
    ModelEvalConfig,
    ModelPusherConfig,
    ModelTrainingConfig,
)

model_training_config = ModelTrainingConfig()
model_training_artifact = ModelTrainingArtifacts(
    model_saved_dir=model_training_config.model_training_artifact_dir,
    model_checkpoint_name=model_training_config.model_checkpoint_name,
)
model_eval_config = ModelEvalConfig()
model_eval_artifact = ModelEvalArtifact(
    is_model_accepted=True,
    changed_accuracy=0.0,
    s3_model_path=model_eval_config.s3_model_dir,
    trained_model_path=model_training_artifact.model_saved_dir,
    model_eval_artifact_dir=model_eval_config.model_evaluation_artifact_dir,
)


model_pusher_config = ModelPusherConfig()
model_pusher = ModelPusher(
    model_eval_artifact=model_eval_artifact, model_pusher_config=model_pusher_config
)
model_pusher_artifact = model_pusher.initiate_model_pusher()
