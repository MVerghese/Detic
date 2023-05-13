import os

def fixformat(read_path,write_path):
	# Write a script that takes in a file and deletes every newline not preceeded by a close bracket
	with open(read_path,'r') as f:
		read_data = f.read()
	# print(repr(read_data[:10000]))
	# print number of times '\n ' is found in read_data
	print(read_data.count("'\n '"))
	read_data = read_data.replace("'\n '","' '")
	print(read_data.count("'\n '"))
	with open(write_path,'w') as f:
		f.write(read_data)

if __name__ == '__main__':
	fixformat('/home/mverghese/PrimitiveDecomp/src/results.txt','/home/mverghese/PrimitiveDecomp/src/results_fixed.txt')