
#%%

## Import libraries
import numpy as np
import matplotlib.pyplot as plt


 
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support,accuracy_score

import argparse

def dataloading( TRAINING_FILE):
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
        
    return X_train, y_train, categories, sampleNames, feature_labels


#%%
    
def perform_plot_LOO(  ):
    #######
    ### Leave-one-samepl-out cross-validation model
    #####

    n_samples, n_features = X_train.shape

    
    y_pred = np.zeros(n_samples)
    class_probs = np.zeros([n_samples,np.unique(y_train).size]) # the probability of assigning each left out sample to each of the classes
    
    loo = LeaveOneOut()
    
    for train_index, test_index in loo.split(X_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        
        clf_main.fit(X_train[train_index,:],y_train[train_index])
        y_pred[test_index] = clf_main.predict(X_train[test_index,:])    
        class_probs[test_index,:] = clf_main.predict_proba(X_train[test_index,:])
    
        
#    my_score = np.mean(y_pred==y_input)
    precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_pred)
    accuracy = accuracy_score(y_train, y_pred)

    ## MAKE CLASS PROBABILITY PLOT
    
    plt.figure()

    arr1inds = y_train.argsort()
    
    labels_train_temp = labels_train.reset_index(drop=True);



    labels_train_sorted = labels_train_temp[arr1inds[::-1]]
    

    prob_loo = class_probs[arr1inds[::-1]]
    #plt.imshow(prob_loo, cmap=plt.cm.coolwarm, interpolation='none', extent=[0,150,0,n_samples])
    plt.imshow(prob_loo, cmap=plt.cm.coolwarm, interpolation='none', aspect='auto'  )
    
    plt.grid(True)
     
    
    plt.yticks(np.arange(n_samples), labels_train_sorted[0:n_samples ], fontsize=2, rotation=0)
    plt.xticks(np.arange( num_categories), categories, fontsize=8 , rotation=45, ha='right')
    ax = plt.gca();
    ax.grid(color='w', linestyle='-', linewidth=0)
    plt.colorbar()
    plt.tight_layout()
    
    outputFile = exportDir + '\\{0}_class_probs_leave_one_out'.format( fname )
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    
#    np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    df = pd.DataFrame( prob_loo, index=labels_train[0:n_samples ], columns=categories )    
    df.to_csv(outputFile + ".csv", index=True, header=True, sep=',')

    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)
        

    ## PRECISION RECALL PLOT
    plotPrecisionRecall( precision, recall, categories, accuracy )
    
    outputFile = exportDir + '\\{0}_precision_recall_training'.format( fname )
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    
    
    data = {'Precision': precision, 'Recall': recall }
    df = pd.DataFrame( data , index=categories )    
    df.to_csv(outputFile + ".csv", index=True, header=True, sep=',')
    
    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)
                
       
          

def plotPrecisionRecall( precision, recall, categories, accuracy ):
    plt.figure()
 
    index = np.arange( len(categories) )
    bar_width = 0.35
    
    for i in range(num_categories):
        plt.bar( index[i], precision[i], bar_width, alpha=1, color=plt.cm.tab20(i) )
        plt.bar( index[i]+bar_width, recall[i], bar_width, alpha=0.5, color=plt.cm.tab20(i) )
#    
#    plt.bar( np.arange(num_categories)-w/2, recall, width=w, align='center' )
#    plt.bar( np.arange(num_categories)+w/2, precision , width=w, align='center', color='blue', alpha=0.7 )
    
    plt.xticks(np.arange(num_categories)+bar_width/2, categories  , rotation=45, ha='right')
    plt.ylabel( 'Parameter Score' )
    plt.tight_layout()
    
    from matplotlib.patches import Patch
    
       
    legend_elements = [ Patch(facecolor=fc, label='Precision'),
                        Patch(facecolor=fc, alpha=0.5, label='Recall') ]
    
    plt.legend(handles=legend_elements, loc='lower right')
    
    
    ax = plt.gca()
    ax.set_title( 'Accuracy: '+str( round(accuracy,2) ) )
    
    plt.tight_layout()

#%%
## Function to obtain feature importances for different models
def getImportantFeatures():
    #K neighbors does not have a way to get most important features. This is a hacky way of getting it
    if args.classifier == 'kneighbors':
        clf_main.fit(X_train, y_train)
        clean_dataset_score = clf_main.score(X_test, y_test)
        
        imp_features = np.zeros(X_train.shape[1])

        for index in range(X_train.shape[1]):
            
            X_train_noisy = X_train.copy()
            np.random.shuffle(X_train_noisy[:, index])
            X_test_noisy = X_test.copy()
            np.random.shuffle(X_test_noisy[:, index])
            clf_main.fit(X_train_noisy, y_train)
            noisy_score = clf_main.score(X_test_noisy, y_test)
#            print(clean_dataset_score - noisy_score, clean_dataset_score, noisy_score)
                
            imp_features[index] = clean_dataset_score - noisy_score
        
    else:
        clf_main.fit(X_train,y_train)
        imp_features = clf_main.feature_importances_ #This is a property of GBC models
    return imp_features

    
def featureImportance( ):
    ## Feature Importance
    
    imp_features = getImportantFeatures()
    
    imp_features_scaled = 100.0 * (imp_features / imp_features.max());
        
    df = pd.DataFrame( imp_features_scaled, index=feature_labels, columns=['Importance'] )    
    
    df = df.sort_values(by='Importance',ascending=False)
    
    df2= df.iloc[0:10]

    plt.figure()


    df2.plot(kind='barh', colormap='jet', legend=False)
    
    outputFile = exportDir + '\\{0}_feature_importance_training'.format( fname )
    
    plt.tight_layout()
    
    plt.xlabel('Relative Importance')
    plt.ylabel('')

    plt.savefig( outputFile + ".png", dpi=dpi_all)
    
#    np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    df.to_csv(outputFile + ".csv", index=True, sep=',')

    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)
        
    ## Two most important Features
    if len(imp_features) > 2:
        from matplotlib.lines import Line2D
        
        if args.blk:
            plt.style.use('dark_background')


        plt.figure()
        plt.tight_layout()

        featureIDX = np.argsort(imp_features)
                    
        for i in range( n_samples):
            plt.plot( X_train[i, featureIDX[-1] ], X_train[i, featureIDX[-2] ], 'o', color=plt.cm.tab20( y_train[i]-1))
    
        plt.xlabel( "Most important PTM (%s) (AVP)"  % feature_labels[featureIDX[-1]])
        plt.ylabel( "Second most important PTM (%s) (AVP)" % feature_labels[featureIDX[-2]])
    
        plt.legend(categories)
        
        legend_elements =[]
        for i in range( num_categories ):
            legend_elements.append( Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i), label=categories[i]) )
            
        plt.legend(handles=legend_elements, loc='best')

        outputFile = exportDir + '\\{0}_firstsec_features_training'.format( fname )
        plt.savefig( outputFile + ".png", dpi=dpi_all)
        
 #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    #    df.to_csv(outputFile + ".csv", index=True, sep=',')
    
        if exportPDF:
            plt.savefig( outputFile + ".pdf", dpi=dpi_all)
            
            
    ## Three most important Features
    if len(imp_features) > 3:
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.lines import Line2D
        
        fig= plt.figure()
        fig.tight_layout()
        
        ax = fig.add_subplot(111, projection='3d')

        featureIDX = np.argsort(imp_features)
        
 

        for i in range( n_samples):
            ax.scatter( X_train[i, featureIDX[-1] ], X_train[i, featureIDX[-2] ], X_train[i, featureIDX[-3] ], 'o', c=plt.cm.tab20( y_train[i]-1))
    
        ax.set_xlabel( "1st PTM (%s) (AVP)"  % feature_labels[featureIDX[-1]])
        ax.set_ylabel( "2nd PTM (%s) (AVP)" % feature_labels[featureIDX[-2]])
        ax.set_zlabel( "3rd PTM (%s) (AVP)" % feature_labels[featureIDX[-3]])
    
        plt.legend(categories)
        
        legend_elements =[]
        for i in range(0, num_categories ):
            legend_elements.append( Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i), label=categories[i]) )
            
        plt.legend(handles=legend_elements, loc='best')

        outputFile = exportDir + '\\{0}_firstsecthird_features_training'.format( fname )
        plt.savefig( outputFile + ".png", dpi=dpi_all)
        
 #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    #    df.to_csv(outputFile + ".csv", index=True, sep=',')
    
        if exportPDF:
            plt.savefig( outputFile + ".pdf", dpi=dpi_all)
    

#%%
###
## Test the model
def perform_plot_Testing( ):
    import math
    

    ## Fit model
    clf_main.fit(X_train, y_train)
    
    y_pred = clf_main.predict(X_test)
    results = clf_main.predict_proba(X_test)
    
    score = clf_main.score(X_test, y_test)
#    print('SCORE: %.2f' % (score))
    
    y_correct = y_pred - y_test
    
    
    
    ########################
    ## Large figure with testing plots 
    
    columns = 4
    rows = (math.ceil( len(y_test)/columns ))


    fig, axs = plt.subplots(rows, columns, figsize=(15, rows*3) )

    axs = axs.ravel()
    fsize = 12

    for i, sample in enumerate(results):
                
        axs[i].bar( np.arange( num_categories ), sample )
        axs[i].set_xticks( np.arange( num_categories )  )
        axs[i].set_xticklabels( categories , rotation=45,ha='right' , fontsize=fsize, fontweight='bold'  )
        axs[i].set_xlabel( '' )
        axs[i].set_ylabel( 'Probability' , fontsize=fsize, fontweight='bold'  )
        axs[i].set_ylim(0, 1)
        for bar in axs[i].patches:
            bar.set_facecolor('red')

        axs[i].patches[y_test[i]-1].set_facecolor('green')



    fig.tight_layout()
    outputFile = exportDir + '\\{0}_testing_plots'.format( fname )
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    
    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)
        
    
    ########################
    ## Plot a single inccorrect testing plot
    
    idx = np.where( y_correct != 0 )

    if( len(idx[0]) ):
        
        i = idx[0][0]
        sample = results[i]
        
        fig, axs = plt.subplots()
    
        axs.bar( np.arange( num_categories ), sample )
        axs.set_xticks( np.arange( num_categories )  )
        axs.set_xticklabels( categories , rotation=45,ha='right' , fontsize=fsize, fontweight='bold'  )
        axs.set_xlabel( '' )
        axs.set_ylabel( 'Probability' , fontsize=fsize, fontweight='bold'  )
        axs.set_ylim(0, 1)
        for bar in axs.patches:
            bar.set_facecolor('red')
    
        axs.patches[y_test[i]-1].set_facecolor('green')
    
    
        fig.tight_layout()
        outputFile = exportDir + '\\{0}_testing_plot_incorrect'.format( fname )
        plt.savefig( outputFile + ".png", dpi=dpi_all)
        
        if exportPDF:
            plt.savefig( outputFile + ".pdf", dpi=dpi_all)


    ########################
    ## Plot a single ccorrect testing plot
    
    idx = np.where( y_correct == 0 )

    if( len(idx[0]) ):
        
        i = idx[0][0]
        sample = results[i]
        
        fig, axs = plt.subplots()
    
        axs.bar( np.arange( num_categories ), sample )
        axs.set_xticks( np.arange( num_categories )  )
        axs.set_xticklabels( categories , rotation=45,ha='right' , fontsize=fsize, fontweight='bold'  )
        axs.set_xlabel( '' )
        axs.set_ylabel( 'Probability' , fontsize=fsize, fontweight='bold'  )
        axs.set_ylim(0, 1)
        for bar in axs.patches:
            bar.set_facecolor('red')
    
        axs.patches[y_test[i]-1].set_facecolor('green')
    
    
        fig.tight_layout()
        outputFile = exportDir + '\\{0}_testing_plot_correct'.format( fname )
        plt.savefig( outputFile + ".png", dpi=dpi_all)
        
        if exportPDF:
            plt.savefig( outputFile + ".pdf", dpi=dpi_all)
            
            
    ########################
    ## Confusion matrix
    from sklearn.metrics import confusion_matrix
    
    cnf_matrix  =  confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(15,8))
    plot_confusion_matrix(cnf_matrix, normalize=True, classes=categories)
    
    outputFile = exportDir + '\\{0}_confusion_matrix_testing'.format( fname )
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    
    
    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)
    
    
    ########################
    ## Precision and Recall 
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    
    plotPrecisionRecall( precision, recall, categories, score )

    outputFile = exportDir + '\\{0}_precision_recall_testing'.format( fname )
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    

    
    data = {'Precision': precision, 'Recall': recall }
    df = pd.DataFrame( data , index=categories )    
    df.to_csv(outputFile + ".csv", index=True, header=True, sep=',')
    
    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)
        
        
    if args.classifier == 'kneighbors':
        ### Plot 2nd 3rd features, test and train sets 
        imp_features = getImportantFeatures()
        
        if len(imp_features) > 2:
            from matplotlib.lines import Line2D
    
            plt.figure()
            plt.tight_layout()
    
            featureIDX = np.argsort(imp_features)
            
            for i in range( n_samples):
                plt.plot( X_train[i, featureIDX[-1] ], X_train[i, featureIDX[-2] ], 'o', color=plt.cm.tab20( y_train[i]-1),
                       markeredgecolor ='k', alpha=0.8)
                
            for i in range( n_test_samples):
                plt.plot( X_test[i, featureIDX[-1] ], X_test[i, featureIDX[-2] ], '*', color=plt.cm.tab20( y_test[i]-1),
                       markeredgecolor ='k', alpha=1, markersize=12 )
                
                
            dist = euclidean_distances(X_train, X_test)
            closest = np.argsort(dist, axis=0)
            
            feat1 = featureIDX[-1]
            feat2 = featureIDX[-2]
        
            for x, neighbors in zip(X_test, closest.T):
                for neighbor in neighbors[:n_neighbors]:
                    try:
                        plt.arrow(x[0,feat1], x[0,feat2], X_train[neighbor, feat1] - x[0,feat1], X_train[neighbor, feat2] - x[0,feat2], head_width=0, fc='k', ec='k', alpha= 0.4)
                    except:
                        print("")
         
            plt.xlabel( "Most important PTM (%s) (AVP)"  % feature_labels[featureIDX[-1]])
            plt.ylabel( "Second most important PTM (%s) (AVP)" % feature_labels[featureIDX[-2]])
        
            plt.legend(categories)
            
            legend_elements =[]
            for i in range( num_categories ):
                legend_elements.append( Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i), label=categories[i]) )
                
            plt.legend(handles=legend_elements, loc='best')
    
    
            outputFile = exportDir + '\\{0}_firstsec_features_testing'.format( fname )
            plt.savefig( outputFile + ".png", dpi=dpi_all)
            
            if exportPDF:
                plt.savefig( outputFile + ".pdf", dpi=dpi_all)

    
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



def compare_Estimators():   
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import LeaveOneOut
        
    # list of (estimator, param_grid), where param_grid is used in GridSearchCV
    classifiers = [
        (LinearSVC(random_state=RS, tol=1e-5, C=0.025)), 
        (KNeighborsClassifier( 3 )),
        (GradientBoostingClassifier(n_estimators=200, random_state=RS, learning_rate = 0.05))]
    
    names = ['LinearSVC', 'KNeighborsClassifier', 'GradientBosstingClassifier']
    
    loo = LeaveOneOut()
    
    
    finalScores = np.zeros( (len(classifiers)) )
    
    # iterate over classifiers
    for i, clf in enumerate(classifiers):               
        finalScores[i] = cross_val_score(clf, X_train, y_train, cv=loo ).mean() 
                       
    plt.figure()
            
    plt.scatter(np.arange(len(finalScores)), finalScores, edgecolors='k')
    plt.xticks(np.arange( len(finalScores)), names)
        
    plt.ylabel( 'Model Name' )
    plt.ylabel( 'Mean accuracy' )
    plt.tight_layout()
    
    
    
    outputFile = '{0}_compare_classifiers_accuracy'.format( fname )
    
    
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    
     #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    #    df.to_csv(outputFile + ".csv", index=True, sep=',')
    
    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)
        
    
def compare_Estimators_fscore():            
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import LeaveOneOut
    
    # list of (estimator, param_grid), where param_grid is used in GridSearchCV
    classifiers = [
        (LinearSVC(random_state=RS, tol=1e-5, C=0.025)), 
        (KNeighborsClassifier( 3 )),
        (GradientBoostingClassifier(n_estimators=200, random_state=RS, learning_rate = 0.05))]
    
    names = ['Linear SVC', 'K-Nearest Neighbors', 'Gradient Boosting']
    
    columns = 3
    fig, axs = plt.subplots(1, columns, figsize=(12, 6) )
    axs = axs.ravel()
    
    loo = LeaveOneOut()
           
    # iterate over classifiers
    for i, clf in enumerate(classifiers):
        
        
        y_pred = np.zeros(n_samples)
       
        for train_index, test_index in loo.split(X_train):
            print("TRAIN:", train_index, "TEST:", test_index)
            
            clf.fit(X_train[train_index,:],y_train[train_index])
            y_pred[test_index] = clf.predict(X_train[test_index,:])    
        
        precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_pred)
        accuracy = accuracy_score(y_train, y_pred)
        
        index = np.arange( num_categories)
        bar_width = 0.3
        
        for c in range(num_categories):
            axs[i].bar( index[c], precision[c], bar_width, alpha=1, color=plt.cm.tab20(i) )
            axs[i].bar( index[c]+bar_width, recall[c], bar_width, alpha=0.6, color=plt.cm.tab20(i) )
            ##axs[i].bar( index[c]+bar_width+bar_width, fscore[c], bar_width, alpha=0.25, color=plt.cm.tab20(i), hatch="//", edgecolor=plt.cm.tab20(i) )
 
                
        axs[i].set_xticks(np.arange(num_categories)+bar_width  )
        axs[i].set_xticklabels( categories , rotation=45, ha='right')
            
            
        axs[i].set_xlabel( '' )
        axs[i].set_ylabel( 'Score' )

        axs[i].set_title( names[i]+ r"$\bf{" + ' | Accuracy: '+ str( round(accuracy,2) ) + "}$" )
        
        
         
    
        plt.tight_layout()
        
        from matplotlib.patches import Patch
        
        legend_elements = [ Patch(facecolor=fc, label='Precision'),
                            Patch(facecolor=fc, alpha=0.6, label='Recall'), ]
                            #Patch(facecolor=fc, alpha=0.25, label='F-score', hatch="//") ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        
    outputFile = exportDir + '\\{0}_compare_classifiers'.format( fname )



    plt.savefig( outputFile + ".png", dpi=dpi_all)
    
     #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
    #    df.to_csv(outputFile + ".csv", index=True, sep=',')
    
    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)
    
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
    
    outputFile = exportDir + '\\{0}_training_deviance'.format( fname )
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    
 #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
#    df.to_csv(outputFile + ".csv", index=True, sep=',')

    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)
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
    
    
    results = euclidean_distances( X, X)
    
    n, m = X.shape
        
    plt.imshow(results, cmap=plt.cm.coolwarm, interpolation='none', aspect='auto'  )
    plt.yticks(np.arange(n), sampleNames[0:n ], fontsize=2, rotation=0)
    plt.grid(True)
     
    plt.yticks(np.arange(n), sampleNames[0:n ], fontsize=2, rotation=0)
    plt.xticks(np.arange(n), sampleNames[0:n ], fontsize=2,  rotation=45,ha='right' )
    
    ax = plt.gca();
    ax.grid(color='w', linestyle='-', linewidth=0)
    plt.colorbar()
 
    outputFile = exportDir + "\\{0}_eucledian_distances".format( fname )
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    
    df = pd.DataFrame( results , index=sampleNames , columns=sampleNames)    
    df.to_csv(outputFile + ".csv", index=True, sep=',')

    if exportPDF:
        plt.savefig( outputFile + ".pdf", dpi=dpi_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNA PTM Classifer script')

    parser.add_argument( '-training', metavar='Training file', help="Specify path to training dataset", required=True)
    
#    parser.add_argument( '-testing', metavar='Testing file', help="Specify path to testing dataset")
    parser.add_argument( '-output', metavar='Output directory', help="Specify output directory")
    parser.add_argument( '-loo', help="Perform leave one out analysis", action='store_true')
    parser.add_argument( '-eucledian', help="Create eucledian distance matrix", action='store_true')
    parser.add_argument( '-features', help="Plot feature importance", action='store_true')
    parser.add_argument( '-all',  help="Performs all analysis included in script", action='store_true')
    parser.add_argument( '-pdf',  help="Exports all figures in PDF format for further editing", action='store_true')
    parser.add_argument( '-model',  help="Model testing plots", action='store_true')
    parser.add_argument( '-random_test',  type= float, help="Select random samples for testing ", required=True)
    parser.add_argument( '-random_state',  type=int, help="Select a random seed for reproducibility")
    parser.add_argument( '-compare_classifiers',  help="Compares some basic classification estimators", action='store_true')
    parser.add_argument( '-classifier',  help="Select classifier to use")
    parser.add_argument( '-test',  help="Perform testing", action='store_true')
    parser.add_argument( '-blk',  help="Exports figures with black background color scheme", action='store_true')
    
    args = parser.parse_args()
    import sys, os


    ### Deals with the output for all files
    if args.output:
        exportDir = os.getcwd() + '//' + args.output        
    else:
        exportDir = os.getcwd()
        
    print("Will export all outputs to %s." % exportDir)
    
    
    fname = args.training.split('.')
    fname = fname[0]
    


    ##Output command as command.txt for future references

    runcmd = " ".join(sys.argv)
    
    file = open( exportDir+ "//command.txt","w") 
    file.write( runcmd   ) 
    file.close() 
    
    
    dpi_all = 500
    
    ## Should i export files in PDF format?
    if args.pdf:
        exportPDF = 1
    else:
        exportPDF = 0

    ## Data loading
    X, y, categories, sampleNames, feature_labels = dataloading( args.training)
    num_categories = len(categories)

    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split( X, y, sampleNames, test_size=args.random_test, random_state= args.random_state, stratify=y)
    n_samples, n_features = X_train.shape
    n_test_samples, n_test_features = X_test.shape
    
    print(labels_train.shape)
    print(labels_test.shape)
    
    print("Training data set: n=%d; Testing dataset: n=%d." % (n_samples, n_test_samples))
    
    ## Random seed
    if args.random_state:
        RS = args.random_state
    else:
        RS = None
        
        
    fc = 'black' ## face color for plot legends
    if args.blk:
        plt.style.use('dark_background')
        fc = 'white'
        
    ## Pick classifier
    if args.classifier == 'gradient':
        params = {'n_estimators': 300, 'learning_rate': 0.05, 'random_state': RS}
        clf_main = GradientBoostingClassifier( **params )
    elif args.classifier == 'kneighbors':
        n_neighbors=3
        clf_main = KNeighborsClassifier( n_neighbors )
   
    if args.loo or args.all:
        perform_plot_LOO()
        
    if args.eucledian or args.all:
        plot_EucledianDistance()
        
    if args.features or args.all:
        featureImportance()
    
    if args.test or args.all:
        perform_plot_Testing( )

    if args.compare_classifiers or args.all:
        compare_Estimators_fscore( )
    
    
