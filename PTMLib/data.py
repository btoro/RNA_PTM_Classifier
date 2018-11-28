
import pandas as pd
import numpy as np

# Splits default PTM dataset into its main components
def splitIntoComponents(df):
    masses = df.iloc[0]
    masses = masses.iloc[2:]
    
    df2 = df.iloc[1:]
    
    ClassLabels  = df2['Class Names'].unique()
    SampleLabels  = df2['Class Names']
    PTMLabels = masses.index
    
    return df2, masses, PTMLabels, SampleLabels, ClassLabels


## This function combines datasets together. Make sure PTM headers are identical
def combineDatasets( df1, df2 ):
    
#    maxclass = df1['Class'].max()
##    
##    uniqueClasses = df2['Class'].unique()
##    
##    newClasses = np.arange(maxclass+1, maxclass+len(uniqueClasses), 1)
##    
##    for i in uniqueClasses:
##        df2['Class'] 
#    
    df3 = pd.concat([df1, df2])
    df3 = df3.fillna(0)
    
    df3['Class Names'] = pd.Categorical( df3['Class Names']  )
    df3['Class'] = df3['Class Names'].cat.codes

    return df3


# This function clears up any PTMS with 0s in all samples. This tends to happen when you break a data set into multiple files
def removeEmptyPTMs( df ):
    df = df.loc[:, (df != 0).any(axis=0)]
    
    return df
