import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_data(training_data, new_data):
    new_data_prepared = new_data.copy()

    # fill missing values with median
    training_data = training_data.fillna({'household_income':training_data.household_income.median(),
                                            'PCR_02':training_data.PCR_02.median()})
    new_data_prepared = new_data_prepared.fillna({'household_income':training_data.household_income.median(),
                                                  'PCR_02':training_data.PCR_02.median()})
    
    
    # scale the data
    scaler_standart = StandardScaler()
    scaler_minmax = MinMaxScaler(feature_range=(-1,1))

    temp_df_train_minmax = training_data[['PCR_01','PCR_03','PCR_04','PCR_06','PCR_08']]
    temp_df_train_standart = training_data[['PCR_02','PCR_05','PCR_07','PCR_09','PCR_10']]
    temp_df_new_minmax = new_data_prepared[['PCR_01','PCR_03','PCR_04','PCR_06','PCR_08']]
    temp_df_new_standart = new_data_prepared[['PCR_02','PCR_05','PCR_07','PCR_09','PCR_10']]

    scaler_minmax.fit(temp_df_train_minmax)
    scaler_standart.fit(temp_df_train_standart)

    new_data_prepared[['PCR_01','PCR_03','PCR_04','PCR_06','PCR_08']] = scaler_minmax.transform(temp_df_new_minmax)
    new_data_prepared[['PCR_02','PCR_05','PCR_07','PCR_09','PCR_10']] = scaler_standart.transform(temp_df_new_standart)

    # add new features
    df_bool = pd.DataFrame({'SpecialProperty' : new_data_prepared["blood_type"].isin(["O+", "B+"])})
    new_data_prepared = pd.concat([new_data_prepared.reset_index(drop=True), df_bool.reset_index(drop=True)], axis=1)
    new_data_prepared.drop(columns=['blood_type'], inplace=True)

    return new_data_prepared