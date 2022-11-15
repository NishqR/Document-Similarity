num_list = list(range(1,128))
#print(num_list)


num_cpus = 5
start_index = 0

for i in range(num_cpus):

	end_index = int((i+1)*(len(num_list)/num_cpus))
	#print(end_index)
	print(len(num_list[start_index:end_index]))
	start_index = int((i+1)*(len(num_list)/num_cpus))