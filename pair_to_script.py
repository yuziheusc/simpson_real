import sys
if __name__ == "__main__":
    #print len(sys.argv)
    if(len(sys.argv)!=3):
        print("Error: Wrong Number of Args.")
        exit()
    fnin = sys.argv[1]
    fnou = sys.argv[2]
    single_vars = set([])
    n_line = 0
    fpou = open(fnou,"w")
    with open("pair.txt") as fpin:
        for line in fpin:
            
            ## skip empty line
            if(line.replace("\n","").replace(" ","")==""):
                continue
            
            n_line += 1
            if(n_line == 1):
                buf = "FILE(%s)"%line.strip().split()[-1]
                print buf
                fpou.write(buf+"\n")
                continue
            
            line = line.strip("\n").split()
            res_var = line[0]
            pre_var = line[1]
            if("," in pre_var):
                single_vars.add((res_var,pre_var.split(",")[0]))
            
            buf = "RES(%s);PRE(%s)"%(res_var, pre_var)
            print buf
            fpou.write(buf+"\n")

            #print single_vars
    for v in single_vars:
        buf = "RES(%s);PRE(%s)"%(v[0], v[1])
        print buf
        fpou.write(buf+"\n")
