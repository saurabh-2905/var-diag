

temp_train1 = (22916, 1229)
temp_train2 = (182692, 32647)
temp_train3 = (403845, 253738)
temp_train4 = (464107, 413401)

mamba_train1 = (95724, 10)
mamba_train2 = (330312, 10)
mamba_train3 = (355993, 10)
mamba_train4 = (591085, 118430)
mamba_train5 = (536435, 356493)
mamba_train6 = (548881, 401074)

contiki_train1 = (599661, 44)
contiki_train2 = (600051, 44)
contiki_train3 = (599912, 44)
contiki_train4 = (597358, 44)
contiki_train5 = (617242, 43)
contiki_train6 = (615058, 44)
contiki_train7 = (595943, 44)
contiki_train8 = (594200, 44)
contiki_train9 = (616138, 44)
contiki_train10 = (342972, 44)
contiki_train11 = (616072, 44)

######## calculate runtime for each application ##########

temp_len = 4
mamba_len = 6
contiki_len = 11

temp_runtime = 0
for i in range(temp_len):
    # print(eval(f'temp_train{i+1}'))
    end_time = eval(f'temp_train{i+1}')[0]
    start_time = eval(f'temp_train{i+1}')[1]
    # print(end_time - start_time)
    temp_runtime += (end_time - start_time)

mamba_runtime = 0
for i in range(mamba_len):
    end_time = eval(f'mamba_train{i+1}')[0]
    start_time = eval(f'mamba_train{i+1}')[1]
    mamba_runtime += (end_time - start_time)

contiki_runtime = 0
for i in range(contiki_len):
    end_time = eval(f'contiki_train{i+1}')[0]
    start_time = eval(f'contiki_train{i+1}')[1]
    contiki_runtime += (end_time - start_time)

print("Temp Sensor Training Runtime (ms):", temp_runtime)
print("MaMBA Training Runtime (ms):", mamba_runtime)
print("Contiki-MAC Training Runtime (ms):", contiki_runtime)
print('')
### Runtime in seconds
print("Temp Sensor Training Runtime (s):", temp_runtime / 1000)
print("MaMBA Training Runtime (s):", mamba_runtime / 1000)
print("Contiki-MAC Training Runtime (s):", contiki_runtime / 1000)
print('')