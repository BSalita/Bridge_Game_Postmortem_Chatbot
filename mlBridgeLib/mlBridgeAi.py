
import pandas as pd
import os
from fastai.tabular.all import nn, load_learner, tabular_learner, cont_cat_split, TabularDataLoaders, TabularPandas, CategoryBlock, RegressionBlock, Categorify, FillMissing, Normalize, EarlyStoppingCallback, RandomSplitter, range_of, MSELossFlat, rmse, accuracy

def train_classifier(df, y_names, cat_names, cont_names, procs=None, valid_pct=0.2, seed=42, bs=1024*5, layers=[512,512,512], epochs=3, device='cuda'):
    splits_ilocs = RandomSplitter(valid_pct=valid_pct, seed=seed)(range_of(df))
    to = TabularPandas(df, procs=procs,
                    cat_names=cat_names,
                    cont_names=cont_names,
                    y_names=y_names,
                    splits=splits_ilocs,
                    #num_workers=10,
                    y_block=CategoryBlock())
    
    # Create a DataLoader
    dls = to.dataloaders(bs=bs, layers=layers, device=device) # cpu or cuda

    # Create a tabular learner
    learn = tabular_learner(dls, metrics=accuracy)

    # Train the model
    learn.fit_one_cycle(epochs) # 1 or 2 epochs is enough to get a good accuracy for large datasets
    return to, dls, learn

def load_data(df, y_names=None, cont_names=None, cat_names=None, procs=None, y_block=None, bs=None, layers=[1024]*4, valid_pct=None, seed=42, max_card=None, device='cuda'):
    """
    Load and preprocess data using FastAI.
    """

    print(f"{y_names=} {cont_names=} {cat_names=} {bs=} {valid_pct=} {max_card=}")
    # Determine number of CPU cores and set workers to cores-1
    num_workers = os.cpu_count() - 1
    print(f"{y_names=} {bs=} {valid_pct=} {num_workers=}")
    if cont_names is not None:
        print(f"{len(cont_names)=} {cont_names=}")
    if cat_names is not None:
        print(f"{len(cat_names)=} {cat_names=}")
    # doesn't work for Contract. assert df.select_dtypes(include=['object','string']).columns.size == 0, df.select_dtypes(include=['object','string']).columns
    assert not df.isna().any().any()
    assert y_names in df, y_names

    # Define continuous and categorical variables
    if cont_names is None and cat_names is None:
        cont_names, cat_names = cont_cat_split(df, max_card=max_card, dep_var=y_names)
        if cont_names is not None:
            print(f"{len(cont_names)=} {cont_names=}")
        if cat_names is not None:
            print(f"{len(cat_names)=} {cat_names=}")
    assert y_names not in [cont_names + cat_names]
    assert set(cont_names).intersection(cat_names) == set(), set(cont_names).intersection(cat_names)
    assert set(cont_names+cat_names+[y_names]).symmetric_difference(df.columns) == set(), set(cont_names+cat_names+[y_names]).symmetric_difference(df.columns)
    assert df[cont_names].select_dtypes(include=['category']).columns.size == 0, df[cont_names].select_dtypes(include=['category']).columns

    # Split the data into training and validation sets
    splits_ilocs = RandomSplitter(valid_pct=valid_pct, seed=seed)(range_of(df))
    #display(df.iloc[splits_ilocs[0]])
    #display(df.iloc[splits_ilocs[1]])
    
    # Load data into FastAI's TabularDataLoaders
    # todo: experiment with specifying a dict of Category types for cat_names: ordinal_var_dict = {'size': ['small', 'medium', 'large']}
    # todo: accept default class of y_block. RegressionBlock for regression, CategoryBlock for classification.

    to = TabularPandas(df, procs=procs,
                    cat_names=cat_names,
                    cont_names=cont_names,
                    y_names=y_names,
                    splits=splits_ilocs,
                    #num_workers=10,
                    y_block=y_block,
                    )
    
    dls = to.dataloaders(bs=bs, layers=layers, device=device) # cpu or cuda

    return dls # return to?

def train_classification(dls, epochs=3, monitor='accuracy', min_delta=0.001, patience=3):
    """
    Train a tabular model for classification.
    """
    print(f"{epochs=} {monitor=} {min_delta=} {patience=}")

    # Create a tabular learner
    learn = tabular_learner(dls, metrics=accuracy)

    # Train the model
    # error: Can't get attribute 'AMPMode' on <module 'fastai.callback.fp16'
    #learn.to_fp16() # to_fp32() or to_bf16()
    
    # Use one cycle policy for training with early stopping
    learn.fit_one_cycle(epochs, cbs=EarlyStoppingCallback(monitor=monitor, min_delta=min_delta, patience=patience)) # sometimes only a couple epochs is optimal
    
    return learn

def train_regression(dls, epochs=20, layers=[200]*10, y_range=(0,1), monitor='valid_loss', min_delta=0.001, patience=3):
    """
    Train a tabular model for regression.
    """
    print(f"{epochs=} {layers=} {y_range=} {monitor=} {min_delta=} {patience=}")
    # todo: check that y_names is numeric, not category.

    learn = tabular_learner(dls, layers=layers, metrics=rmse, y_range=y_range, loss_func=MSELossFlat()) # todo: could try loss_func=L1LossFlat.

    # Use mixed precision training. slower and error.
    # error: Can't get attribute 'AMPMode' on <module 'fastai.callback.fp16'
    #learn.to_fp16() # to_fp32() or to_bf16()
    
    # Use one cycle policy for training with early stopping
    learn.fit_one_cycle(epochs, cbs=EarlyStoppingCallback(monitor=monitor, min_delta=min_delta, patience=patience)) # todo: experiment with using lr_max?
    
    return learn

def save_model(learn, f):
    learn.export(f)

def load_model(f):
    return load_learner(f)

def get_predictions(learn, data, device='cpu'):
    data[learn.dls.train.x_names].info(verbose=True)
    data[learn.dls.train.y_names].info(verbose=True)
    assert set(learn.dls.train.x_names).difference(data.columns) == set(), f"df is missing column names which are in the model's training set:{set(learn.dls.train.x_names).difference(data.columns)}"
    dl = learn.dls.test_dl(data, device=device)
    probs, actual = learn.get_preds(dl=dl)
    return probs, actual

def predictions_to_df(data, y_names, preds):
    """
    Create a DataFrame with actual and predicted values.
    """
    
    df = pd.DataFrame({
        f'{y_names}_Actual': data[y_names],
        f'{y_names}_Pred': preds,
     })

    if data[y_names].dtype == 'category':
        df[f'{y_names}_Match'] = data[y_names] == preds
    else:
        df[f'{y_names}_Diff'] = data[y_names] - preds 

    df = pd.concat([df, data.drop(columns=[y_names])], axis='columns')
    
    return df

def make_predictions(f, data):
    """
    Make predictions using a trained tabular model.
    """
    learn = load_learner(f)
    return get_predictions(learn, data)

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
