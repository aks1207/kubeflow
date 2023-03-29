import kfp
from kfp import components
from kfp.components import InputPath, OutputPath

# Define the components
train_op = components.load_component_from_file('train_op.yaml')
predict_op = components.load_component_from_file('predict_op.yaml')

# Define the pipeline
@kfp.dsl.pipeline(
    name='Boston House Price Prediction',
    description='A pipeline that predicts Boston house prices using XGBoost'
)
def boston_house_price_prediction(
    data_path: InputPath(),
    model_path: OutputPath(),
    predictions_path: OutputPath()
):
    # Train the model
    model = train_op(
        data_path=data_path,
        model_path=model_path
    ).output

    # Make predictions
    predictions = predict_op(
        data_path=data_path,
        model_path=model,
        predictions_path=predictions_path
    )

# Define the default values for the inputs
DATA_PATH = 'data/boston_housing.csv'
MODEL_PATH = 'model/xgboost.model'
PREDICTIONS_PATH = 'predictions/predictions.csv'

# Define the arguments for the pipeline
arguments = {
    'data_path': DATA_PATH,
    'model_path': MODEL_PATH,
    'predictions_path': PREDICTIONS_PATH
}

# Compile and run the pipeline
kfp.compiler.Compiler().compile(boston_house_price_prediction, 'boston_house_price_prediction.tar.gz')
kfp.Client().create_run_from_pipeline_func(boston_house_price_prediction, arguments=arguments)
