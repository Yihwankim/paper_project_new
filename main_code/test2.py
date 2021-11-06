# import packages
import pandas as pd
import numpy as np

########################################################################################################################
df_data = pd.read_excel('data_process/conclusion/NN/nn_index_data_interaction.xlsx', header=0, skipfooter=0)
X_with = df_data.copy()

# sample set to array
X_with_test = np.array(X_with)

index_pred = model1.predict(X_with_test)
df_with = pd.DataFrame(index_pred)