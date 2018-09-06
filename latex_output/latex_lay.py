if __name__ == "__main__":

    target_var = "NULL"
    r2_agg_list = {}
    r2_dis_list = {}
    r2_ipv_list = {} # r2 improved, between dis and agg. 
    r2_multi_list = {}
    k_agg = {}
    k_multi = {}
    pvalue_agg = {}
    pvalue_multi = {}
    
    pair_list = []
    with open("pair.txt","r") as fpin1:
        for _ in range(2):
            next(fpin1)
        for line in fpin1:
            line = line.strip("\n").split()
            print line
            preds = line[1]
            pred_var,pred_cond = preds.split(",")
            r2_agg = float(line[2])
            r2_dis = float(line[3])
            r2_ipv = float(line[4])
            #print "simpson#", pred_var, pred_cond, r2_agg, r2_dis
            r2_agg_list[(pred_var, pred_cond)] = r2_agg
            r2_dis_list[(pred_var, pred_cond)] = r2_dis
            r2_ipv_list[(pred_var, pred_cond)] = r2_ipv
            pair_list += [(pred_var, pred_cond, r2_ipv)]
    with open("data_output.txt") as fpin2:
        for line in fpin2:
            line = line.strip("\n").split()
            print line
            target_var = line[1]
            n_preds = int(line[0])
            pred_var = line[2]
            pred_cond = "NULL"
            if(n_preds == 2):
                pred_cond = line[3]
                
            k,p,r2 = None, None, None
            if(n_preds == 2):
                r2 = float(line[4])
                k = float(line[6])
                p = float(line[9])
                pvalue_multi[(pred_var, pred_cond)] = p
                k_multi[(pred_var, pred_cond)] = k
                r2_multi_list[(pred_var, pred_cond)] = r2
            else:
                r2 = float(line[3])
                k = float(line[5])
                p = float(line[7])
                pvalue_agg[(pred_var, pred_cond)] = p
                k_agg[(pred_var, pred_cond)] = k
                
            
            print "multireg#", pred_var, pred_cond, r2, k, p


    ## first sort the pars.
    pair_list.sort(key = lambda x : (x[0], -x[2], x[1]))
    #print pair_list
    
    ## ouptut the data as a latex table
    with open("table_out.txt", "w") as fpou:
        buf = "\\begin{table}[H]" + "\n" + "\centering"
        print buf
        fpou.write(buf+"\n")
        
        buf = "\\begin{tabular}{ | p{5cm} | c | c | c | c | c | c | c | c | }"
        print buf
        fpou.write(buf+"\n")

        
        buf = "  Simpson's Pair & $\\beta_1$(Agg) & $\\beta_1$(Mul)& $\\Delta \\beta_1$  & p-value(Agg)& p-value(Mul)& $R\_{\\text{AGG}}^2$ & $R_{\\text{Mul}}^2$& $R_{\\text{DIS}}^2$\\\\"
        buf = "\\hline" + "\n" + buf
        print buf
        fpou.write(buf+"\n")
        
        var_current = "NULL"
        for entry in pair_list:
            pred_var = entry[0]
            pred_cond = entry[1]
            pair = (entry[0], entry[1])
            buf = "  %s, %s &  %6.4g & %6.4g & %6.4g\\%%  & %6.4g & %6.4g & %6.4f & %6.4f & %6.4f \\\\"%(pred_var.replace("_","\\_"), pred_cond.replace("_","\\_"), k_agg[(pred_var,"NULL")], k_multi[pair], (k_multi[pair]/k_agg[(pred_var,"NULL")] - 1)*100 , pvalue_agg[(pred_var,"NULL")], pvalue_multi[pair], r2_agg_list[pair], r2_multi_list[pair], r2_dis_list[pair])
            if(var_current != pred_var):
                var_current = pred_var
                buf = "  \\hline"+"\n"+buf
            print buf
            fpou.write(buf+"\n")

        buf = "  \\hline"
        print buf
        fpou.write(buf+"\n")


        buf = "\\end{tabular}"
        print buf
        fpou.write(buf+"\n")

        
        buf = "\\end{table}"
        print buf
        fpou.write(buf+"\n")
                
        
        ## output the plots for aggregated, disaggregated fitting.
    with open("regression_out.txt", "w") as fpou:
        var_current = "NULL"
        for entry in pair_list:
            pred_var = entry[0]
            pred_cond = entry[1]
            pair = (entry[0], entry[1])
            fname1 = pred_var + "-vs-" + pred_cond + ".pdf"
            fname2 = target_var + "_vs_" + "(" + pred_var + "+" + pred_cond + ")"
            buf = "\n\\begin{figure}[H]"
            print buf
            fpou.write(buf+"\n")
            buf = "  \\centering"
            print buf
            fpou.write(buf+"\n")
            buf = "  \\includegraphics[page=3, width=0.45\\textwidth]{output/%s}"%(fname1)
            print buf
            fpou.write(buf+"\n")
            buf = "  \\includegraphics[page=7, width=0.45\\textwidth]{output/%s}"%(fname1)
            print buf
            fpou.write(buf+"\n")
            buf = "\\caption{Simpson's pairs. Variable [%s] condition on [%s]. Regression, Aggregated, Disaggregated.}"%(pred_var.replace("_","\\_"), pred_cond.replace("_","\\_"))
            print buf
            fpou.write(buf+"\n")
            buf = "\\end{figure}"
            print buf
            fpou.write(buf+"\n")

        
    ## output residual
    with open("residual_out.txt", "w") as fpou:
        var_current = "NULL"
        for entry in pair_list:
            pred_var = entry[0]
            pred_cond = entry[1]
            pair = (entry[0], entry[1])
            fname1 = pred_var + "-vs-" + pred_cond + ".pdf"
            fname2 = target_var + "_vs_" + "(" + pred_var + "+" + pred_cond + ")"
            buf = "\n\\begin{figure}[H]"
            print buf
            fpou.write(buf+"\n")
            buf = "  \\centering"
            print buf
            fpou.write(buf+"\n")
            buf = "  \\includegraphics[page=4, width=0.30\\textwidth]{output/%s}"%(fname1)
            print buf
            fpou.write(buf+"\n")
            buf = "  \\includegraphics[page=9, width=0.30\\textwidth]{output/%s}"%(fname1)
            print buf
            fpou.write(buf+"\n")
            buf = "  \\includegraphics[page=1, width=0.30\\textwidth]{multireg/%s}"%(fname2)
            print buf
            fpou.write(buf+"\n")
            buf = "  \caption{Residual plot. Variable [%s] condition on [%s]. Aggregated, Disaggregated and multiple regression.}"%(pred_var.replace("_","\\_"), pred_cond.replace("_","\\_"))
            print buf
            fpou.write(buf+"\n")
            buf = "\\end{figure}"
            print buf
            fpou.write(buf+"\n")


