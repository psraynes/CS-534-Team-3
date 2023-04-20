import csv
from load_image import *
        
file_name = input("Please name the output file")
loaded_data = load_all_files()
csv_file = open(file_name,'w')
csv_writer = csv.writer(csv_file)
header = ["uid","h","s","v","con1","cor1","con2","cor2","con3","cor3","con4","cor4","label"]
csv_writer.writerow(header)
csv_writer.writerows(loaded_data)

