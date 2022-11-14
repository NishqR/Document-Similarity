num_list = list(range(1,181))
#print(num_list)


num_cpus = 4
start_index = 0

for i in range(num_cpus):

	end_index = int((i+1)*(len(num_list)/num_cpus))
	print(end_index)
	print(num_list[start_index:end_index])
	start_index = int((i+1)*(len(num_list)/num_cpus))