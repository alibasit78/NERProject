from ner.components.model_pusher import ModelPusher
from ner.entity.artifact_entity import ModelTrainingArtifacts
from ner.entity.config_entity import (
    ModelPusherConfig,
    ModelTrainingConfig,
)

model_training_config = ModelTrainingConfig()
model_training_artifact = ModelTrainingArtifacts(
    model_saved_dir=model_training_config.model_training_artifact_dir,
    model_checkpoint_name=model_training_config.model_checkpoint_name,
)

model_pusher_config = ModelPusherConfig()
model_pusher = ModelPusher(
    model_training_artifact=model_training_artifact, model_pusher_config=model_pusher_config
)
model_pusher_artifact = model_pusher.initiate_model_pusher()
