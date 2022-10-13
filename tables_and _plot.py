import pickle
import csv

class Create:
            
    def save_avg_global_ratio(self,data):
        #store glocal ratio after each episode
        ep = list(data)[-1]
        file_name = "global_ratio_ep_"+str(ep)
        with open(file_name, "wb") as f:
            pickle.dump(data[ep],f)
        
    def q_table(self,data,ep):
        for k in data.keys():
          row = data[k]
          file_name = 'q_table_antID_'+str(k)+'_ep_'+str(ep)+'.csv'
          with open(file_name, 'w') as f: 
              write = csv.writer(f) 
              write.writerows(row) 
