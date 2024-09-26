# from ner.pipline.training_pipeline import TrainPipeline
# train_pipeline = TrainPipeline()
# train_pipeline.run_pipeline()

from ner.pipline.prediction_pipeline import ModelPredictor

model_predictor = ModelPredictor()
input_sentence = "John stays in Unites states of America with his wife Jerry"
model_predictor.initiate_model_predicctor(input_sentence=input_sentence)
