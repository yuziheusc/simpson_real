import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats import kurtosis

def do_regression(data_frame, response_var, predicators):
    Y = np.array(data_frame[response_var])
    X = np.array(data_frame[predicators])
    X = sm.add_constant(X)
    linear_model = sm.OLS(Y,X)
    lr = linear_model.fit()

    ## make a plot
    fnou = response_var+"_vs_"+"("+("+".join(predicators))+")"+".pdf"
    pp = PdfPages(fnou)
    if(len(predicators) == 1):
        ## plot the data points
        plt.clf()
        plt.scatter(X[:,1], Y[:], s=0.5, c='b')
        ## plot the fitting line
        x1 = np.arange(min(X[:,1]), max(X[:,1]), (max(X[:,1])-min(X[:,1]))*0.01 )
        y1 = x1 * lr.params[1] + lr.params[0]
        plt.plot(x1,y1,'b--')
        plt.xlabel(predicators[0])
        plt.ylabel(response_var)
        plt.title(response_var+"_vs_"+predicators[0])
        
        pp.savefig(bbox_inches='tight', papertype='a4')
    
        
        ## plot the residule-x
        plt.clf()
        plt.scatter(X[:,1], lr.resid, s = 5.0, c='b', alpha=0.4, linewidth=0.0)
        plt.xlabel(predicators[0])
        plt.ylabel("Residual")
        pp.savefig(bbox_inches='tight', papertype='a4')

        ## plot residual square-x
        plt.clf()
        plt.scatter(X[:,1], lr.resid**2, s = 5.0, c='b', alpha=0.4, linewidth=0.0)
        plt.xlabel(predicators[0])
        plt.ylabel("Residual Square")
        pp.savefig(bbox_inches='tight', papertype='a4')
        
    y_pred = lr.predict(X)
    
    ## plot the residule-y-pred
    plt.clf()
    plt.scatter(y_pred, lr.resid, s = 5.0, c='r', alpha=0.4, linewidth=0.0)
    plt.xlabel(response_var)
    plt.ylabel("Residual")
    pp.savefig(bbox_inches='tight', papertype='a4')

    ## plot residual square-y-pred
    plt.clf()
    plt.scatter(y_pred, lr.resid**2, s = 5.0, c='r', alpha=0.4, linewidth=0.0)
    plt.xlabel(response_var)
    plt.ylabel("Residual Square")
    pp.savefig(bbox_inches='tight', papertype='a4')

    ## plot observed-predicted
    plt.clf()
    plt.scatter(y_pred, Y, s = 5.0, c='g', alpha=0.4, linewidth=0.0)
    y1 = np.arange(min(Y),max(Y),(max(Y)-min(Y))*0.01)
    plt.plot(y1,y1,'b--')
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    #plt.title("Aggregated Regression")
    plt.axis('equal')
    pp.savefig(bbox_inches='tight', papertype='a4')
    

    ## make a histogram of residual
    plt.clf()
    plt.hist(lr.resid, bins=40)
    plt.title("Regression Residual Distribution")
    res_hist = np.histogram(lr.resid,bins=40)
    bin_w = res_hist[1][1] - res_hist[1][0]

    res_mean = np.mean(lr.resid)
    res_std = np.std(lr.resid)
    x1 = np.arange(min(lr.resid), max(lr.resid), (max(lr.resid)-min(lr.resid))*0.01 )
    y1 = norm.pdf((x1-res_mean)/res_std)*len(lr.resid)*bin_w/res_std
    plt.plot(x1,y1,'b--')
    
    pp.savefig(bbox_inches='tight', papertype='a4')
    
    
    pp.close()
    
    print("fitting parameters : ")
    for i in range(len(lr.params)):
        print("  %4d  %6.4g"%(i,lr.params[i]))
    
    print("standard error : ")
    for i in range(len(lr.bse)):
        print("  %4d  %6.4g"%(i,lr.bse[i]))
        
    print("p-value : ")
    for i in range(len(lr.pvalues)):
        print("  %4d  %6.4g"%(i,lr.pvalues[i]))
    
    #print(lr.ssr)
    print("r-square : ", lr.rsquared)

    print("** residual analysis **")
    print("skewness of residual : %6.4g"%(skew(lr.resid)))
    print("kurtosis of residual : %6.4g"%(kurtosis(lr.resid)))
    
    print("** BreuschPagan test **")
    test = sms.het_breushpagan(lr.resid, lr.model.exog)
    name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
    for i in range(len(test)):
        print("%s : %5.3g"%(name[i],test[i]))

    print("** Normality of residual **")
    k2, p = stats.normaltest(lr.resid)
    print("p-value = ", p)

    ## write a line of record to the output data file
    fpou = open("data_output.txt","a")
    
    buf = "%d %s %s"%(len(predicators), response_var, " ".join(predicators) )
    buf = buf + " %6.4g"%(lr.rsquared)
    for i in range(len(lr.params)):
        buf = buf + " %6.4g"%(lr.params[i])
    for i in range(len(lr.pvalues)):
        buf = buf + " %6.4g"%(lr.pvalues[i])
    fpou.write(buf+"\n")
    fpou.close()
    
        
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("ERROR: Wrong number of args.")
        sys.exit()
    fnin = sys.argv[1]
    print("\nSTART\n")
    print("Script file : %s"%(fnin))

    ## create the empty output data file.
    fpou = open("data_output.txt","w")
    fpou.close()

    try:
        fpin = open(fnin)
    except IOError:
        print("ERROR: No script file found.")
        sys.exit()
    with fpin:
        #fp_data = None
        data_frame = None
        for line in fpin:
            line = line.lstrip()
            line = line.strip('\n')
            if(line == ""): continue
            if(line[0] == "#"): continue
            #print("-",line,"-")

            ## data file description
            if(line[0:4]=="FILE"):
                fn_data = line[4:].lstrip(" (").rstrip(" )")
                print("\nOpen data file :", fn_data)
                data_frame = pd.read_csv(fn_data)
                
            ## regression description
            if(line[0:3]=="RES"):
                line = line.split(';')
                
                ## filter the response and predicator list
                for i in range(2):
                    line[i] = line[i][3:].replace('(',' ').replace(')',' ').replace(',',' ')
                    line[i] = line[i].lstrip().rstrip()
                    line[i] = line[i].split(' ')
                
                
                if(len(line[0])!=1):
                    print("Error: One and only one response var is allowed.")
                    sys.exit()
                    
                response_var = line[0][0]
                predicators = line[1]
                #print("Regression on ['%s'] with predicators %s"%(response_var, predicators) )
                print("\nRegression: ")
                print("Response : ['%s']"%(response_var))
                print("Predicators : %s"%(predicators))

                do_regression(data_frame, response_var, predicators)
                
                print("\n")

    print("\nEND.")
