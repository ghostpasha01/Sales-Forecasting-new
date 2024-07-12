from ml_pipeline import utils, processing, train, test
import projectpro
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# 1. Read Training Data, Seperate Test data will be set aside for future inference in part 2
data = utils.read_data("data/train_kOBLwZA.csv")

# 2. Process and encode data
df = processing.preprocess_data(data)
projectpro.checkpoint('362665')


# 3. Training data will be split data into train and test sets
X_train, X_test, y_train, y_test = utils.split_data(df, 0.2, 22222)

# 4. Model Training on various models
model_gbm, model_gam = train.train_models(X_train, y_train, 3, 5, 20)

# 5. Predictions on validation set splitted from training data
y_pred_gbm, y_pred_gam = test.predictions(X_test, y_test, model_gbm, model_gam)
projectpro.checkpoint('362665')
