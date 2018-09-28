
#%%

## Import libraries
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import precision_recall_fscore_support
 
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
 
import scipy.io as sio
 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split

import argparse

def dataloading( TRAINING_FILE, TESTING_FILE):
    ####
    ## Training data set:
    ####
    train_data = pd.read_csv( TRAINING_FILE, sep=',', delimiter=None, header=None)
    
    ### Get labels
    feature_labels = train_data.drop([0,1], axis=1)
    feature_labels = feature_labels.iloc[0]
    feature_labels = feature_labels.reset_index(drop=True)
    
    ### Get Sample Names
    sampleNames = train_data.iloc[:,0]
    sampleNames = sampleNames.iloc[2:]
    sampleNames = sampleNames.reset_index(drop=True)
    
    ### Set up data arrays for training
    
    X_train = train_data.fillna(0)
    
    
    # The first and second column in data is the class names, which is removed from X_train. This is y_train.
    X_train=X_train.iloc[2:,2:] 
    X_train= X_train.reset_index(drop=True )
    X_train.columns = range(X_train.shape[1])
    X_train=np.asmatrix(X_train).astype(float)
    
    
    
    y_train=train_data.iloc[2:,1] # second column of data
    y_train = y_train.fillna(0)
    
    y_train=y_train.astype(int).ravel()
    
    n_samples, n_features = X_train.shape
     
    
    ## Determine category names based on sample names.
    unique = np.unique(y_train, return_index=1 )
    
    categories = []
    
    for item in unique[1]:
        categories.append( sampleNames[item] )
        
    ####
    ## Testing data set:
    ####
    
    if TESTING_FILE:
        
        test_data = pd.read_csv(TESTING_FILE, sep=',', delimiter=None, header=None)
        
        ## Also remove first column
        X_test = test_data 
        X_test = X_test.fillna(0)
        
        # The first and second column in data is the class names, which is removed from X_train. This is y_train.
        X_test=X_test.iloc[2:,2:] 
        X_test= X_test.reset_index(drop=True )
        #X_test.columns = range(X_train.shape[1])
        X_test=np.asmatrix(X_test).astype(float)
        
        
        
        y_test=test_data.iloc[2:,1] # second column of data
        y_test = y_test.fillna(0)
        
        y_test=y_test.astype(int).ravel()
    
    
        return X_train, y_train, X_test, y_test, categories, sampleNames, feature_labels
    else:
        return X_train, y_train, None, None, categories, sampleNames, feature_labels


#%%
    
def perform_plot_LOO():
    #######
    ### Leave-one-samepl-out cross-validation model
    #####

    n_samples, n_features = X_train.shape

    
    y_pred = np.zeros(n_samples)
    imp_features = np.zeros(n_features)
    class_probs = np.zeros([n_samples,np.unique(y_train).size]) # the probability of assigning each left out sample to each of the classes
    
    loo = LeaveOneOut()
    
    for train_index, test_index in loo.split(X_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        
        clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.05) 
        clf.fit(X_train[train_index,:],y_train[train_index])
        imp_features = imp_features + clf.feature_importances_
        y_pred[test_index] = clf.predict(X_train[test_index,:])    
        class_probs[test_index,:] = clf.predict_proba(X_train[test_index,:])
    
        
    my_score = np.mean(y_pred==y_train)
    precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_pred)
    
    ## MAKE CLASS PROBABILITY PLOT
    
    plt.figure()

    prob_loo = class_probs
    #plt.imshow(prob_loo, cmap=plt.cm.coolwarm, interpolation='none', extent=[0,150,0,n_samples])
    plt.imshow(prob_loo, cmap=plt.cm.coolwarm, interpolation='none', aspect='auto'  )
    
    plt.grid(True)
     
    
    plt.yticks(np.arange(n_samples), sampleNames[0:n_samples ], fontsize=2, rotation=0)
    plt.xticks(np.arange( num_categories), categories, fontsize=8 , rotation=90)
    ax = plt.gca();
    ax.grid(color='w', linestyle='-', linewidth=0)
    plt.colorbar()
    plt.tight_layout()
    
    outputFile = exportDir + '\\class_probs_leave_one_out'
    plt.savefig( outputFile + ".png", dpi=120)
    
#    np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    df = pd.DataFrame( prob_loo, index=sampleNames[0:n_samples ], columns=categories )    
    df.to_csv(outputFile + ".csv", index=True, header=True, sep=',')

    if exportEPS:
        plt.savefig( outputFile + ".eps", dpi=120)
        

    ## Precision and Recall 
    
    plt.figure()
    
    w = 0.5
    
    index = np.arange( num_categories)
    bar_width = 0.35
    
    for i in range(num_categories):
        plt.bar( index[i], precision[i], bar_width, alpha=1, color=plt.cm.tab10(i) )
        plt.bar( index[i]+bar_width, recall[i], bar_width, alpha=0.5, color=plt.cm.tab10(i) )
#    
#    plt.bar( np.arange(num_categories)-w/2, recall, width=w, align='center' )
#    plt.bar( np.arange(num_categories)+w/2, precision , width=w, align='center', color='blue', alpha=0.7 )
    
    plt.xticks(np.arange(num_categories)+bar_width/2, categories  , rotation=90)
    plt.ylabel( 'Parameter Score' )
    plt.tight_layout()
    
    from matplotlib.patches import Patch
    
    legend_elements = [ Patch(facecolor='black', label='Precision'),
                        Patch(facecolor='black', alpha=0.5, label='Recall') ]
    plt.legend(handles=legend_elements, loc='best')
    

    
    
    outputFile = exportDir + '\\precision_recall'
    plt.savefig( outputFile + ".png", dpi=120)
    
    
    data = {'Precision': precision, 'Recall': recall }
    df = pd.DataFrame( data , index=categories )    
    df.to_csv(outputFile + ".csv", index=True, header=True, sep=',')
    
    if exportEPS:
        plt.savefig( outputFile + ".eps", dpi=120)
                
       
          

     
#%%
## Create the model with the entire data set


def featureImportance( imp_features ):
    ## Feature Importance
    
    imp_features_scaled = 100.0 * (imp_features / imp_features.max());
        
    df = pd.DataFrame( imp_features_scaled, index=feature_labels, columns=['Importance'] )    
    
    df = df.sort_values(by='Importance',ascending=False)
    
    df2= df.iloc[0:10]

    plt.figure()


    df2.plot(kind='barh', colormap='jet', legend=False)
    
    outputFile = exportDir + '\\feature_importance'
    
    plt.tight_layout()
    
    plt.xlabel('Relative Importance')
    plt.ylabel('')

    plt.savefig( outputFile + ".png", dpi=120)
    
#    np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    df.to_csv(outputFile + ".csv", index=True, sep=',')

    if exportEPS:
        plt.savefig( outputFile + ".eps", dpi=120)
        
        
    ## Two most important Features
    if len(imp_features) > 2:
        from matplotlib.lines import Line2D

        plt.figure()
        plt.tight_layout()

        featureIDX = np.argsort(imp_features)
        
 

        for i in range( n_samples):
            plt.plot( X_train[i, featureIDX[-1] ], X_train[i, featureIDX[-2] ], 'o', color=plt.cm.tab10( y_train[i]-1))
    
        plt.xlabel( "Most important PTM (%s) (AVP)"  % feature_labels[featureIDX[-1]])
        plt.ylabel( "Second most important PTM (%s) (AVP)" % feature_labels[featureIDX[-2]])
    
        plt.legend(categories)
        
        legend_elements =[]
        for i in range( num_categories ):
            legend_elements.append( Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i), label=categories[i]) )
            
        plt.legend(handles=legend_elements, loc='best')

        outputFile = exportDir + '\\firstsec_features'
        plt.savefig( outputFile + ".png", dpi=120)
        
 #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    #    df.to_csv(outputFile + ".csv", index=True, sep=',')
    
        if exportEPS:
            plt.savefig( outputFile + ".eps", dpi=120)
            
            
    ## Three most important Features
    if len(imp_features) > 3:
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D
        
        fig= plt.figure()
        fig.tight_layout()
        
        ax = fig.add_subplot(111, projection='3d')

        featureIDX = np.argsort(imp_features)
        
 

        for i in range( n_samples):
            ax.scatter( X_train[i, featureIDX[-1] ], X_train[i, featureIDX[-2] ], X_train[i, featureIDX[-3] ], 'o', color=plt.cm.tab10( y_train[i]-1))
    
        ax.set_xlabel( "1st PTM (%s) (AVP)"  % feature_labels[featureIDX[-1]])
        ax.set_ylabel( "2nd PTM (%s) (AVP)" % feature_labels[featureIDX[-2]])
        ax.set_zlabel( "3rd PTM (%s) (AVP)" % feature_labels[featureIDX[-3]])
    
        plt.legend(categories)
        
        legend_elements =[]
        for i in range( num_categories ):
            legend_elements.append( Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i), label=categories[i]) )
            
        plt.legend(handles=legend_elements, loc='best')

        outputFile = exportDir + '\\firstsecthird_features'
        plt.savefig( outputFile + ".png", dpi=120)
        
 #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    #    df.to_csv(outputFile + ".csv", index=True, sep=',')
    
        if exportEPS:
            plt.savefig( outputFile + ".eps", dpi=120)
    


def createModel( ):
    
    clf = GradientBoostingClassifier( **params ) 
    clf.fit(X_train,y_train)
    
    imp_features = clf.feature_importances_
    
    return clf, imp_features



#%%
###
## Test the model
def perform_plot_Testing( ):
    import math

    clf = GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    
#    y_pred = clf.predict(X_test)
    results = clf.predict_proba(X_test)
    
    score = clf.score(X_test, y_test)
    print('SCORE: %.2f' % (score))
    
    columns = 4
    rows = (math.ceil( len(y_test)/columns ))

    
    fig, axs = plt.subplots(rows, columns, figsize=(15, rows*3) )

    axs = axs.ravel()

    for i, sample in enumerate(results):
        

        
#        predictedCategory = categories[y_pred[i]-1]
        

        
        axs[i].bar( np.arange( num_categories ), sample )
        axs[i].set_xticks( np.arange( num_categories )  )
        axs[i].set_xticklabels( categories , rotation=90   )
        
        axs[i].set_xlabel( '' )
        axs[i].set_ylabel( 'Probability' )
        for bar in axs[i].patches:
            bar.set_facecolor('red')

        axs[i].patches[y_test[i]-1].set_facecolor('green')

#        
#        if( y_pred[i] != y_test[i] ):
#            axs[i].set_title( 'Predicted: '+predictedCategory+'. Actual: '+categories[y_test[i]-1], color='red'  )
#        else:
#            axs[i].set_title( 'Predicted: '+predictedCategory+'. Actual: '+categories[y_test[i]-1], color='green'  )
        fig.tight_layout()

    outputFile = exportDir + '\\testing_plots'
    plt.savefig( outputFile + ".png", dpi=120)
    
 #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
#    df.to_csv(outputFile + ".csv", index=True, sep=',')

    if exportEPS:
        plt.savefig( outputFile + ".eps", dpi=120)


    
## Gradient boosting regularization
##http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regularization.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regularization-py
def perform_Regularization( ):
    plt.figure(figsize=(12, 6))


    original_params = {'n_estimators': 500, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}

    for label, color, setting in [('No shrinkage', 'orange',
                                   {'learning_rate': 1.0, 'subsample': 1.0}),
                                  ('learning_rate=0.1', 'turquoise',
                                   {'learning_rate': 0.1, 'subsample': 1.0}),
#                                  ('subsample=0.5', 'blue',
#                                   {'learning_rate': 1.0, 'subsample': 0.5}),
                                  ('learning_rate=0.1, subsample=0.5', 'gray',
                                   {'learning_rate': 0.1, 'subsample': 0.5}),
                                  ('learning_rate=0.1, max_features=2', 'magenta',
                                   {'learning_rate': 0.1, 'max_features': 2}),
                                   ('learning_rate=0.05', 'red',
                                   {'learning_rate': 0.05, 'max_leaf_nodes': None, 'min_samples_split': 2 })]:
        params = dict(original_params)
        params.update(setting)
    
        clf = GradientBoostingClassifier(**params)
        clf.fit(X_train, y_train)

        # compute test set deviance
        test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
        
        
        for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
            test_deviance[i] = clf.loss_(y_test, y_pred)
            
#        train_deviance = clf.train_score_
    
        plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
                '-', color=color, label=label)
    
    plt.legend(loc='upper left')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Test Set Deviance')
    
    outputFile = exportDir + '\\training_deviance'
    plt.savefig( outputFile + ".png", dpi=120)
    
 #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
#    df.to_csv(outputFile + ".csv", index=True, sep=',')

    if exportEPS:
        plt.savefig( outputFile + ".eps", dpi=120)
#    fig = plt.figure()
#    
#    X_test=np.asmatrix(X_test)
#    
#    y_pred = clf.predict(X_test)
#    results = clf.predict_proba(X_test)
#    
#    for idx, sample in enumerate(results):
#        fig = plt.figure()
#        
#        predictedCategory = categories[y_pred[idx]-1]
#        
#        plt.bar( np.arange(7), sample )
#        plt.xticks(np.arange(7), categories,  rotation=90)
#        
#        plt.xlabel( 'Categories' )
#        plt.ylabel( 'Probability' )
#        
#        if( y_pred[idx] != Y_test_true[idx] ):
#            plt.title( 'Predicted: '+predictedCategory+'. Actual: '+categories[Y_test_true[idx]-1], color='red'  )
#        else:
#            plt.title( 'Predicted: '+predictedCategory+'. Actual: '+categories[Y_test_true[idx]-1], color='green'  )
#    
#    
#        plt.show

#%%

##########
## Eucledian Distance Calculation & Plot
    
def plot_EucledianDistance():

    plt.figure()
    
    
    results = euclidean_distances( X_train, X_train)
    
    
    plt.imshow(results, cmap=plt.cm.coolwarm, interpolation='none', aspect='auto'  )
    plt.yticks(np.arange(n_samples), sampleNames[0:n_samples ], fontsize=2, rotation=0)
    plt.grid(True)
     
    plt.yticks(np.arange(n_samples), sampleNames[0:n_samples ], fontsize=2, rotation=0)
    plt.xticks(np.arange(n_samples), sampleNames[0:n_samples ], fontsize=2, rotation=90)
    
    ax = plt.gca();
    ax.grid(color='w', linestyle='-', linewidth=0)
    plt.colorbar()
 
    outputFile = exportDir + '\\eucledian_distances'
    plt.savefig( outputFile + ".png", dpi=120)
    
    df = pd.DataFrame( results , index=sampleNames , columns=sampleNames)    
    df.to_csv(outputFile + ".csv", index=True, sep=',')

    if exportEPS:
        plt.savefig( outputFile + ".eps", dpi=120)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNA PTM Classifer script')

    parser.add_argument( '-training', metavar='Training file', help="Specify path to training dataset", required=True)
    
    parser.add_argument( '-testing', metavar='Testing file', help="Specify path to testing dataset")
    parser.add_argument( '-output', metavar='Output directory', help="Specify output directory")
    parser.add_argument( '-loo', help="Perform leave one out analysis", action='store_true')
    parser.add_argument( '-eucledian', help="Create eucledian distance matrix", action='store_true')
    parser.add_argument( '-features', help="Plot feature importance", action='store_true')
    parser.add_argument( '-all',  help="Performs all analysis included in script", action='store_true')
    parser.add_argument( '-eps',  help="Exports all figures in EPS format for further editing", action='store_true')
    parser.add_argument( '-model',  help="Model testing plots", action='store_true')
    parser.add_argument( '-random_test',  type= float, help="Select random samples for testing ")
    parser.add_argument( '-random_state',  help="Select random samples for testing ", action='store_true')

    
    args = parser.parse_args()
    
    if args.output:
        exportDir = args.output        
    else:
        import os
        exportDir = os.getcwd()
        
    print("Will export all outputs to %s." % exportDir)

    if args.eps:
        exportEPS = 1
    else:
        exportEPS = 0


    X_train, y_train, X_test, y_test, categories, sampleNames, feature_labels = dataloading( args.training, args.testing )
    num_categories = len(categories)
    
   
    params = {'n_estimators': 1000, 'learning_rate': 0.05}

    if args.random_test:
        X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=args.random_test, random_state=2)
        n_samples, n_features = X_train.shape
        n_test_samples, n_test_features = X_test.shape
        
        
        print("Training data set: n=%d; Testing dataset: n=%d." % (n_samples, n_test_samples))

        
        perform_plot_Testing( )


    if args.loo or args.all:
        perform_plot_LOO()
        
    if args.eucledian or args.all:
        plot_EucledianDistance()
        
    if args.features or args.all:
        clf_model, imp_features = createModel()
        featureImportance(imp_features)
    
    if args.testing:
        clf_model, imp_features = createModel()
        perform_plot_Testing( )
    
    if args.model:
        perform_Regularization ( )
    
    
