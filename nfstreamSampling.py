from nfstream import NFStreamer
import pandas as pd
import argparse
import numpy as np
from datetime import datetime

def Sampling(cap):
    stream = NFStreamer(source=cap, accounting_mode=3, statistical_analysis=True)
    info = stream.to_pandas()[["id", "requested_server_name", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "src2dst_packets", 
                               "src2dst_bytes", "dst2src_packets", "dst2src_bytes","bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
                               "bidirectional_duration_ms","bidirectional_packets", "bidirectional_bytes", "bidirectional_max_piat_ms", "bidirectional_mean_piat_ms", "bidirectional_stddev_piat_ms", "bidirectional_stddev_ps",
                               "bidirectional_rst_packets", "bidirectional_fin_packets"]]
    
    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input file')
    parser.add_argument('-d', '--delta', nargs='?',required=False, help='samplig delta interval (sec)',default=1)
    parser.add_argument('-s', '--sni', nargs='?',required=False, help='API SNI',default="")
    args = parser.parse_args()

    deltams = float(args.delta)*1000
    sni = args.sni

    output_file = ''.join(args.input.split('.')[:-1])+"_features.dat"

    data = Sampling(args.input)

    data = data.sort_values(by=['bidirectional_first_seen_ms'])

    win = np.where(data['bidirectional_first_seen_ms'] < 1694013408122)

    if sni != "":
        data = data[data['requested_server_name'] == sni]
    
    Ti = data.iloc[0]['bidirectional_first_seen_ms']
    Tf = data.iloc[-1]['bidirectional_first_seen_ms']
    print("First timestamp: {} - {}".format(Ti,datetime.utcfromtimestamp(Ti/1000)))
    print("Last timestamp: {} - {}".format(Tf,datetime.utcfromtimestamp(Tf/1000)))

    t = Ti
    obs = 0
    while t < Tf-deltams:
        cond1 = data['bidirectional_first_seen_ms'] > t
        cond2 = data['bidirectional_first_seen_ms'] < t+deltams
        win = np.where(cond1&cond2)
        
        nflows = data.iloc[win]['id'].count()

        avgupbytes = data.iloc[win]['dst2src_bytes'].mean()
        stdupbytes = data.iloc[win]['dst2src_bytes'].std()

        avgdownbytes = data.iloc[win]['src2dst_bytes'].mean()
        stddownbytes = data.iloc[win]['src2dst_bytes'].std()

        avgduration = data.iloc[win]["bidirectional_duration_ms"].mean()
        stdduration = data.iloc[win]["bidirectional_duration_ms"].std()

        avgratio = avgdownbytes/avgupbytes if avgupbytes > 0 else avgdownbytes
        stdratio = stddownbytes/stdupbytes if stdupbytes > 0 else stddownbytes

        

        nfinflags = data.iloc[win]["bidirectional_fin_packets"].sum()
        nrstflags = data.iloc[win]["bidirectional_rst_packets"].sum()
        #print(nflows,avgupbytes,stdupbytes,avgdownbytes,stddownbytes,avgduration,stdduration,avgratio,stdratio)

        t += deltams
        obs += 1

        if nflows > 0:
            f = np.nan_to_num(np.array([nflows, avgupbytes, stdupbytes, avgdownbytes, stddownbytes, avgduration, stdduration, avgratio, stdratio, nfinflags, nrstflags]))
            print(nflows,avgupbytes,stdupbytes,avgdownbytes,stddownbytes,avgduration,stdduration,avgratio,stdratio)
            if 'allfeatures' not in locals():
                    allfeatures = f.copy()
            else:
                    allfeatures = np.vstack((allfeatures, f))

    np.savetxt(output_file, allfeatures, fmt='%.4f')
    
    


if __name__ == '__main__':
    main()