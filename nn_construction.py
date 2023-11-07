from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor

import joblib

# Read in dataset created by the PID for training and testing, of the structure
# ['x', 'y', 'psi', 'u', 'v', 'r', 'e_psi', 'cmd_rpm','act_rpm','cmd_rudder','act_rudder']
S = pd.read_csv("../data file/heading_data.csv")

# Separate data to be used for training and testing
# X (input)
rudder_data_list = S.iloc[0:-1, [4, 5, 6, 7]]  # u,v,r, e_psi in m/s, m/s, rad, rad
# Y (output)
rudder_label_list = S.iloc[1:, 10]  # rudder command in degrees

# Shuffle and split dataset for training and testing
X2, y2 = shuffle(rudder_data_list.values, rudder_label_list, random_state=0)
test_size = 0.2
x2_train, x2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=test_size)

# Create NN and normalize data
mlp_psi = make_pipeline(
    StandardScaler(),
    MLPRegressor(solver='adam', hidden_layer_sizes=(20, 20), activation='relu', max_iter=1000),
)

# Train NN on the training data
mlp_psi.fit(x2_train, y2_train.ravel())

# Test trained NN on the test data
print('mlp_psi.score: ', mlp_psi.score(x2_test, y2_test))

# Save the model so it can be called in a separate script
joblib.dump(mlp_psi, '../saved_model/mlp_psi.pkl')

#dataset and model for rpm
# rpm_data_list = S.iloc[0:-1, [9]] # u,v,r,e_pos_x, e_pos_y, e_psi
# rpm_label_list = S.iloc[1:, 8]  # rpm, actual shaft speed
# X1, y1 = shuffle(rpm_data_list.values, rpm_label_list, random_state=0)
#
# x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=test_size)
# print(x1_train.shape)
# print(x1_test.shape)
#
# mlp1 = make_pipeline(
#     StandardScaler(),
#     MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000),
# )
#
# mlp1.fit(x1_train, y1_train.ravel())
# print('mlp1.score: ', mlp1.score(x1_test, y1_test))

# joblib.dump(mlp_rpm, '../saved_model/mlp_rpm.pkl')

