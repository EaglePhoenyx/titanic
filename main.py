"""

"""

from src.titanic.data import load_data,clean_data,prepare_data
from src.titanic.registry import save_model, load_model
from src.titanic.train import train_model, evaluate_model, optimize_model
from src.titanic.utils import hello_world

print(hello_world())

train_df, test_df = load_data()

print(train_df.shape, test_df.shape)

train_df_cleaned = clean_data(train_df)
test_df_cleaned = clean_data(test_df)

print(train_df_cleaned.shape, test_df_cleaned.shape)

(X_train, y_train) = prepare_data(train_df_cleaned)
(X_test, y_test) = prepare_data(test_df_cleaned)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = train_model(X_train,y_train)

""" print(X_test.isna().sum())
print(y_test.isna().sum()) """

print(model,model.coef_)
evaluate_model(model,X_test,y_test)

optimized_model = optimize_model(model,X_train,y_train)
print('Optimized model')
evaluate_model(optimized_model,X_test,y_test)

save_model(optimized_model, './models/model.pkl')

model_tryload = load_model('./models/model.pkl')
print(model_tryload)

print('Loaded model')
evaluate_model(model_tryload)