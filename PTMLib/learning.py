import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dpi_all = 500
fc = 'k'

## Test the model
def learning_testing( clf, X_train, y_train, X_test, y_test, classes , exportName  ):
    import math
    

    ## Fit model
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    results = clf.predict_proba(X_test)

    score = clf.score(X_test, y_test)
#    print('SCORE: %.2f' % (score))
    
    columns = 4
    rows = (math.ceil( len(y_test)/columns ))

    fig, axs = plt.subplots(rows, columns, figsize=(15, rows*3) )

    axs = axs.ravel()
    fsize = 12
    
    y_test = y_test.reset_index( drop =True)

    for i, sample in enumerate(results):

        axs[i].bar( np.arange( classes.shape[0] ), sample )
        axs[i].set_xticks( np.arange( classes.shape[0] )  )
        axs[i].set_xticklabels( classes , rotation=90 , fontsize=fsize, fontweight='bold'  )
        axs[i].set_xlabel( '' )
        axs[i].set_ylabel( 'Probability' , fontsize=fsize, fontweight='bold'  )
        for bar in axs[i].patches:
            bar.set_facecolor('red')

        ind=int(y_test.iloc[i])-1
        axs[i].patches[ind].set_facecolor('green')



    fig.tight_layout()
    outputFile = exportName + '_testing_plots'
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    plt.savefig( outputFile + ".eps", dpi=dpi_all)


    ## Confusion matrix
    from sklearn.metrics import confusion_matrix
    
    cnf_matrix  =  confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #########################
    plt.figure(figsize=(15,8))
    plot_confusion_matrix(cnf_matrix, normalize=True, classes=classes)
    
    outputFile = exportName + '_confusion_matrix_testing'
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    plt.savefig( outputFile + ".eps", dpi=dpi_all)
    
    ## Precision and Recall 
    #############################
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

    plt.figure()
 
    index = np.arange( classes.shape[0] )
    bar_width = 0.35
    
    for i in range( classes.shape[0] ):
        plt.bar( index[i], precision[i], bar_width, alpha=1, color=plt.cm.tab20(i) )
        plt.bar( index[i]+bar_width, recall[i], bar_width, alpha=0.5, color=plt.cm.tab20(i) )


    plt.xticks(np.arange( classes.shape[0] )+bar_width/2, classes  , rotation=90)
    plt.ylabel( 'Parameter Score' )
    
    from matplotlib.patches import Patch
    
    legend_elements = [ Patch(facecolor=fc, label='Precision'),
                        Patch(facecolor=fc, alpha=0.5, label='Recall') ]
    plt.legend(handles=legend_elements, loc='best')
    
    ax = plt.gca()

    ax.text(0.95, 0.9, ('%.2f' % score).lstrip('0'), size=15,
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),
                transform=ax.transAxes, horizontalalignment='right')
    
    plt.tight_layout()

    outputFile = exportName + '_precision_recall_testing'
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    plt.savefig( outputFile + ".eps", dpi=dpi_all)
    

#    
#    data = {'Precision': precision, 'Recall': recall }
#    df = pd.DataFrame( data , index=classes )    
#    df.to_csv(outputFile + ".csv", index=True, header=True, sep=',')
#
    
    
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
    plt.xticks(tick_marks, classes, rotation=90)
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


def GBC_featurePlot( clf, X_train, y_train, feature_labels, categories, exportName):
    ## Feature Importance
    
    clf.fit(X_train,y_train)
    imp_features = clf.feature_importances_ #This is a property of GBC models    
    
    imp_features_scaled = 100.0 * (imp_features / imp_features.max());
        
    
    df = pd.DataFrame( imp_features_scaled, index=feature_labels, columns=['Importance'] )    
    
    
#    print(df )


    df = df.sort_values(by='Importance',ascending=False)
    
    df2= df.iloc[0:10]

    plt.figure()


    df2.plot(kind='barh', colormap='jet', legend=False)
    

    
    
    
    plt.tight_layout()
    
    plt.xlabel('Relative Importance')
    plt.ylabel('')

   
    outputFile = exportName + '_feature_importance_training'
    plt.savefig( outputFile + ".png", dpi=dpi_all)
    plt.savefig( outputFile + ".eps", dpi=dpi_all)
    
#        
    ## Two most important Features
    if len(imp_features) > 2:
        from matplotlib.lines import Line2D
        
#        if args.blk:
#            plt.style.use('dark_background')

        try:
            feature_labels = X_train.columns
        except:
            X_train = pd.DataFrame( X_train )
            feature_labels = X_train.columns

        plt.figure()

        featureIDX = np.argsort(imp_features)
                    
        for i in range( X_train.shape[0] ):
            plt.plot( X_train.iloc[i, featureIDX[-1] ], X_train.iloc[i, featureIDX[-2] ], 'o', c=plt.cm.tab20( int(y_train.iloc[i]) -1) )
    
        plt.xlabel( "Most important PTM (%s) (AVP)"  % feature_labels[featureIDX[-1]])
        plt.ylabel( "Second most important PTM (%s) (AVP)" % feature_labels[featureIDX[-2]])
    
        plt.legend(categories)
        
        legend_elements =[]
        for i in range( categories.shape[0] ):
            legend_elements.append( Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i), label=categories[i]) )
            
        plt.legend(handles=legend_elements, loc='best')
 
    
        plt.tight_layout()

        outputFile = exportName + '_firstsec_features_training'
        plt.savefig( outputFile + ".png", dpi=dpi_all)
        plt.savefig( outputFile + ".eps", dpi=dpi_all) 
            
#    ## Three most important Features
#    if len(imp_features) > 3:
#        
#        from mpl_toolkits.mplot3d import Axes3D
#        from matplotlib.lines import Line2D
#        
#        fig= plt.figure()
#        fig.tight_layout()
#        
#        ax = fig.add_subplot(111, projection='3d')
#
#        featureIDX = np.argsort(imp_features)
#        
# 
#
#        for i in range( n_samples):
#            ax.scatter( X_train[i, featureIDX[-1] ], X_train[i, featureIDX[-2] ], X_train[i, featureIDX[-3] ], 'o', c=plt.cm.tab20( y_train[i]-1))
#    
#        ax.set_xlabel( "1st PTM (%s) (AVP)"  % feature_labels[featureIDX[-1]])
#        ax.set_ylabel( "2nd PTM (%s) (AVP)" % feature_labels[featureIDX[-2]])
#        ax.set_zlabel( "3rd PTM (%s) (AVP)" % feature_labels[featureIDX[-3]])
#    
#        plt.legend(categories)
#        
#        legend_elements =[]
#        for i in range(0, num_categories ):
#            legend_elements.append( Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i), label=categories[i]) )
#            
#        plt.legend(handles=legend_elements, loc='best')
#
#        outputFile = exportDir + '\\firstsecthird_features_training'
#        plt.savefig( outputFile + ".png", dpi=dpi_all)
#        
# #       np.savetxt(outputFile + ".csv", prob_loo, delimiter=",", header=",".join(categories ))
#    #    df.to_csv(outputFile + ".csv", index=True, sep=',')
#    
#        if exportEPS:
#            plt.savefig( outputFile + ".eps", dpi=dpi_all)
    
