
from fastai.tabular.all import *

def sanity_check_data(data, y_name):
    assert not data.isna().any().any()
    assert 'bool' not in data.dtypes
    assert y_name in data, y_name

def load_data(data, y_name):
    """
    Load and preprocess data using FastAI.
    """
    # Define continuous and categorical variables
    cont, cat = cont_cat_split(data, max_card=20, dep_var=y_name)
    
    # Determine number of CPU cores and set workers to cores-1
    num_workers = os.cpu_count() - 1
    
    # Load data into FastAI's TabularDataLoaders
    dls = TabularDataLoaders.from_df(data, y_names=y_name, y_block=RegressionBlock,
                                     cat_names=cat, cont_names=cont, procs=[Categorify, FillMissing, Normalize],
                                     valid_idx=list(range(int(0.8*len(data)), len(data))), bs=4096, num_workers=num_workers)
    
    return dls

def train_model(dls, y_name, epochs=10):
    """
    Train a tabular model using FastAI.
    """
    learn = tabular_learner(dls, layers=[1024]*6, metrics=rmse)
    
    # Use one cycle policy for training with early stopping
    learn.fit_one_cycle(epochs, cbs=EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=5))
    
    return learn

def save_model(learn, f):
    learn.export(f)

def load_model(f):
    return load_learner(f)

def get_predictions(learn, data):
    dl = learn.dls.test_dl(data)
    preds, _ = learn.get_preds(dl=dl)
    return preds.squeeze().numpy()

def predictions_to_df(data, y_name, preds):
    """
    Create a DataFrame with actual and predicted values.
    """
    
    df = pd.DataFrame({
        f'{y_name}_Actual': data[y_name],
        f'{y_name}_Pred': preds,
        f'{y_name}_Diff': data[y_name] - preds
    })
    
    df = pd.concat([df, data.drop(columns=[y_name])], axis='columns')
    
    return df

def make_predictions(f, data):
    """
    Make predictions using a trained tabular model.
    """
    learn = load_learner(f)
    data[learn.dls.train.x_names].info(verbose=True)
    data[learn.dls.train.y_names].info(verbose=True)
    dl = learn.dls.test_dl(data)
    preds, _ = learn.get_preds(dl=dl)
    return preds.squeeze().numpy()

# obsolete pytorch stuff for app.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import TensorDataset, DataLoader
# import sklearn # only needed to get __version__
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import safetensors # only needed to get __version__
# from safetensors.torch import load_file, save_file
# import pickle

# Create a Model
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, output_size, hidden_layer_sizes=[1024]*6):
#         super(NeuralNetwork, self).__init__()
        
#         # Create a list of sizes representing each layer (input + hidden + output)
#         all_sizes = [input_size] + hidden_layer_sizes + [output_size]
        
#         # Dynamically create the linear layers
#         self.layers = nn.ModuleList([
#             nn.Linear(all_sizes[i], all_sizes[i+1]) for i, _ in enumerate(all_sizes[:-1])
#         ])
        
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Process input through each linear layer followed by ReLU, except the last layer
#         for layer in self.layers[:-1]:
#             x = self.relu(layer(x))
        
#         # Last layer is followed by sigmoid
#         x = self.sigmoid(self.layers[-1](x))
#         return x

    # with open(predicted_rankings_model_file,'rb') as f:
    #     y_name, columns_to_scale, X_scaler, y_scaler = pickle.load(f) # on lenovo5-1tb, had to manually copy pickle file from /mnt/e/bridge because git pull seems to have pulled a bad copy.

    # predicted_rankings_model_filename = f"acbl_{club_or_tournament}_predicted_rankings_model.pth"
    # predicted_rankings_model_file = savedModelsPath.joinpath(predicted_rankings_model_filename)
    # if not predicted_rankings_model_file.exists():
    #     st.error(f"Oops. {predicted_rankings_model_filename} not found.")
    #     return None
    # with open(predicted_rankings_model_file,'rb') as f:
    #     model_state_dict = torch.load(f, map_location=torch.device('cpu'))

    # print('y_name:', y_name, 'columns_to_scale:', columns_to_scale)
    # st.session_state.df.info(verbose=True)
    # assert set(columns_to_scale).difference(set(st.session_state.df.columns)) == set(), set(columns_to_scale).difference(set(st.session_state.df.columns))

    # df = st.session_state.df.copy()
    # df['Date'] = pd.to_datetime(df['Date']).astype('int64') # only need to do once (all rows have same value) then assign to all rows.
    # for d in mlBridgeLib.NESW:
    #     df['Player_Number_'+d] = pd.to_numeric(df['Player_Number_'+d], errors='coerce').astype('float32').fillna(0) # float32 because could be NaN
    # df['Vul'] = df['Vul'].astype('uint8') # 0-3
    # df['Dealer'] = df['Dealer'].astype('category')

    # df = df[[y_name]+columns_to_scale.tolist()].copy() # todo: columns_to_scale needs to be made a list before saving to pkl
    # assert df.isna().sum().sum() == 0, df.columns[df.isna().sum().gt(0)] # todo: must be a better way of showing columns with na.
    # X = df.drop(columns=[y_name])
    # y = df[y_name]
    # for col in X.select_dtypes(include='category').columns:
    #     X[col] = X[col].cat.codes
    # assert X.select_dtypes(include=['category','string']).empty

    # X_scaled = X.copy()
    # X_scaled = X_scaler.transform(X)

    # # Initialize the model and load weights
    # model_for_pred = NeuralNetwork(X_scaled.shape[1],1) # 1 is output_size
    # model_for_pred.load_state_dict(model_state_dict)

    # # Make predictions
    # model_for_pred.eval()
    # with torch.no_grad():
    #     predictions_scaled = model_for_pred(torch.tensor(X_scaled, dtype=torch.float32)) # so fast (1ms) that we're good with using the CPU

    # predicted_board_result_ns = y_scaler.inverse_transform(predictions_scaled)
    # predicted_board_result_ns_s = pd.Series(predicted_board_result_ns.flatten())
    # predicted_board_result_ns_adjusted_s = predicted_board_result_ns_s*.5/predicted_board_result_ns_s.mean() # scale predictions so NS mean is 50%
    # # Create a DataFrame for predictions and save or further use
