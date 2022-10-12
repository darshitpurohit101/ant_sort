import matplotlib.pyplot as plt
import csv

class Create:
            
    def plot_avg_global_ratio(self,data):
        ep = list(data)[-1]
        y = [i for i in range(len(data[ep]))]
        plt.plot(y,data[ep], label = "ep1")
        plt.xlabel("Iterations")
        plt.ylabel("Average ratio for same color")
        plt_name = "avg_global_ratio_ep_"+str(ep)
        plt.savefig(plt_name)
        
    def q_table(self,data,ep):
        for k in data.keys():
          row = data[k]
          file_name = 'q_table_antID_'+str(k)+'_ep_'+str(ep)+'.csv'
          with open(file_name, 'w') as f: 
              write = csv.writer(f) 
              write.writerows(row) 
