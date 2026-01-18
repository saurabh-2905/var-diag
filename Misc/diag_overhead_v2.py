import matplotlib.pyplot as plt

# APP = 'mamba' 
APP = 'contiki'

plt.rcParams.update({
    "font.size": 18,          # default text size
    "axes.titlesize": 18,     # title size
    "axes.labelsize": 18,     # x/y labels
    "xtick.labelsize": 18,    # tick labels
    "ytick.labelsize": 18,
    "legend.fontsize": 18,    # legend text
})

# Data from the table
# if APP == 'mamba':
#     events = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000]
#     dust = [6557, 6568, 6762, 6689, 6645, 9175, 13576, 13496, 15162]
#     varl = [9882, 10681, 11829, 12440, 13613, 18923, 25020, 35400, 46010]
# elif APP == 'contiki':
#     events = [1000, 2000, 3000, 4000, 5000, 10000, 15000]
#     dust = [1262, 5298, 9128, 9229, 13442, 26345, 33180]
#     varl = [2061, 2432, 2750, 3390, 3930, 6732, 10104]

# events_m = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000]
# dust_m = [6557, 6568, 6762, 6689, 6645, 9175, 13576, 13496, 15162]
# varl_m = [9882, 10681, 11829, 12440, 13613, 18923, 25020, 35400, 46010]
# events_c = [1000, 2000, 3000, 4000, 5000, 10000, 15000]
# dust_c = [1262, 5298, 9128, 9229, 13442, 26345, 33180]
# varl_c = [2061, 2432, 2750, 3390, 3930, 6732, 10104]

############ VarDiag contiki data ############
# Reference data length
vd_contiki_reference_data_length = [700, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 12000, 15000]

# Training time taken (ms)
vd_contiki_training_time_ms = [118, 323, 613, 1060, 1650, 2044, 3040, 4540, 5981, 7786]
# Testing time per event trace (ms)
vd_contiki_testing_time_ms = [47, 123, 277, 385, 702, 801, 1161, 1633, 2063, 2551]
# F1 score
vd_contiki_f1_score = [0.76, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92]

############ VarDiag Habitat data ############
# Reference data length
vd_habitat_reference_data_length = [700, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 12000, 15000, 17000, 20000, 22000, 25000]
# Training time taken (ms)
vd_habitat_training_time_ms = [106, 284, 942, 1598, 2332, 3265, 4582, 8445, 11787, 13836, 18783, 24364, 29495, 34965]
# Testing time per event trace (ms)
vd_habitat_testing_time_ms = [175, 434, 971, 1563, 2645, 2460, 3307, 4780, 5548, 6129, 7531, 8900, 9809, 11097]
# F1 score
vd_habitat_f1_score = [0.5, 0.51, 0.56, 0.58, 0.58, 0.71, 0.71, 0.73, 0.74, 0.77, 0.78, 0.78, 0.78, 0.79]

############## VarDiag Temp-Sensor data ##############  
# Reference data length
vd_temp_reference_data_length = [700, 1000, 2000]

# Training time taken (ms)
vd_temp_training_time_ms = [106, 133, 274]

# Testing time per event trace (ms)
vd_temp_testing_time_ms = [177, 220, 347]

# F1 score
vd_temp_f1_score = [0.96, 0.96, 0.96]


################ Dustminer Contiki data ##############
# Reference data length
dus_contiki_reference_data_length = [700, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 12000, 15000]

# Training time taken (ms)
dus_contiki_training_time_ms = [733, 727, 4762, 8448, 8404, 12664, 18516, 25603, 30186, 33042]

# Testing time taken (ms)
dus_contiki_testing_time_ms = [403, 407, 408, 402, 400, 407, 407, 441, 406, 645]

# F1 score
dus_contiki_f1_score = [0.19] * 10

################# dustminer habitat data ##############
# Reference data length
dus_habitat_reference_data_length = [700, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 12000, 15000, 17000, 20000, 22000, 25000]

# Training time taken (ms)
dus_habitat_training_time_ms = [2211, 2212, 2200, 2265, 2256, 2386, 4613, 4590, 5889, 9095, 9017, 8982, 9797, 10651]

# Testing time taken (ms)
dus_habitat_testing_time_ms = [4268, 4471, 4335, 4239, 4205, 4215, 4235, 4258, 4282, 4257, 4303, 4188, 4327, 4341]

# F1 score
dus_habitat_f1_score = [0.24] * 14

################# dustminer temp-sensor data ##############
# Reference data length
dus_temp_reference_data_length = [700, 1000, 2000]

# Training time taken (ms)
dus_temp_training_time_ms = [41, 55, 99]

# Testing time per event trace (ms)
dus_temp_testing_time_ms = [207, 210, 289]

# F1 score
dus_temp_f1_score = [0.4, 0.4, 0.4]

# # Plot setup
# plt.figure(figsize=(8, 7))
# plt.plot(events_m, dust_m, 'o-', label='Dustminer-habitat', linewidth=3, markersize=8, color="#A6290D")
# plt.plot(events_m, varl_m, 's--', label='VarDiag-habitat', linewidth=3, markersize=8, color="#A6290D")
# plt.plot(events_c, dust_c, 'o-', label='Dustminer-contiki', linewidth=3, markersize=8, color="#4F7302")
# plt.plot(events_c, varl_c, 's--', label='VarDiag-contiki', linewidth=3, markersize=8, color="#4F7302")

# # Labels and title
# plt.xlabel('No. of Events (Reference Data)', fontsize=20)
# plt.ylabel('Time (ms)', fontsize=20)
# plt.title('End-to-end Computation Overhead', fontsize=20)
# plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=2)
# plt.grid(True, linestyle='--', alpha=0.6)

# # Optionally use logarithmic scale if the difference is large
# # plt.yscale('log')

# Create figure and first y-axis
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10, 10), sharex=True)

ax1.plot(vd_temp_reference_data_length, vd_temp_training_time_ms, marker='o', markersize=6,linestyle='-' ,label='Train. VarDiag Smart-Build.', color="#0367A6")
ax2.plot(vd_temp_reference_data_length, vd_temp_testing_time_ms, marker='o', markersize=6, linestyle='-', label='Test. VarDiag Smart-Build.', color="#0367A6", alpha=1)

ax1.plot(vd_habitat_reference_data_length, vd_habitat_training_time_ms, marker='o', markersize=6,linestyle='-' ,label='Train. VarDiag Habitat', color="#A6290D")
ax2.plot(vd_habitat_reference_data_length, vd_habitat_testing_time_ms, marker='o', markersize=6, linestyle='-', label='Test. VarDiag Habitat', color="#A6290D", alpha=1)

ax1.plot(vd_contiki_reference_data_length, vd_contiki_training_time_ms, marker='o', markersize=6,linestyle='-' ,label='Train. VarDiag Contiki', color="#4F7302")
ax2.plot(vd_contiki_reference_data_length, vd_contiki_testing_time_ms, marker='o', markersize=6, linestyle='-', label='Test. VarDiag Contiki', color="#4F7302", alpha=1)

ax1.plot(dus_temp_reference_data_length, dus_temp_training_time_ms, marker='o', markersize=6,linestyle='--' ,label='Train. Dustminer Smart-Build.', color="#0367A6")
ax2.plot(dus_temp_reference_data_length, dus_temp_testing_time_ms, marker='o', markersize=6, linestyle='--', label='Test. Dustminer Smart-Build.', color="#0367A6", alpha=1)

ax1.plot(dus_habitat_reference_data_length, dus_habitat_training_time_ms, marker='o', markersize=6,linestyle='--' ,label='Train. Dustminer Habitat', color="#A6290D")
ax2.plot(dus_habitat_reference_data_length, dus_habitat_testing_time_ms, marker='o', markersize=6, linestyle='--', label='Test. Dustminer Habitat', color="#A6290D", alpha=1)

ax1.plot(dus_contiki_reference_data_length, dus_contiki_training_time_ms, marker='o', markersize=6,linestyle='--' ,label='Train. Dustminer Contiki', color="#4F7302")
ax2.plot(dus_contiki_reference_data_length, dus_contiki_testing_time_ms, marker='o', markersize=6, linestyle='--', label='Test. Dustminer Contiki', color="#4F7302", alpha=1)


# ax1.set_xlabel('Reference Data Length')
# ax1.set_xscale('log')
ax1.set_ylabel('Time (ms)')
ax1.grid(True)
ax1.legend(loc='center left', fontsize=11, bbox_to_anchor=(1.0, 0.5), fancybox=True, ncol=1)

ax2.set_ylabel('Time (ms)')
ax2.grid(True)
ax2.legend(loc='center left', fontsize=11, bbox_to_anchor=(1.0, 0.5), fancybox=True, ncol=1)


ax3.plot(vd_temp_reference_data_length, vd_temp_f1_score, linestyle='-', label='F1 VarDiag Smart-Build.', color="#0367A6", marker='o', markersize=6) #  

ax3.plot(vd_habitat_reference_data_length, vd_habitat_f1_score, linestyle='-', label='F1 VarDiag Habitat', color="#A6290D", marker='o', markersize=6) #  

ax3.plot(vd_contiki_reference_data_length, vd_contiki_f1_score, linestyle='-', label='F1 VarDiag Contiki', color="#4F7302", marker='o', markersize=6) #  

ax3.plot(dus_temp_reference_data_length, dus_temp_f1_score, linestyle='--', label='F1 Dustminer Smart-Build.', color="#0367A6", marker='o', markersize=6) #  

ax3.plot(dus_habitat_reference_data_length, dus_habitat_f1_score, linestyle='--', label='F1 Dustminer Habitat', color="#A6290D", marker='o', markersize=6) #  

ax3.plot(dus_contiki_reference_data_length, dus_contiki_f1_score, linestyle='--', label='F1 Dustminer Contiki', color="#4F7302", marker='o', markersize=6) #  

ax3.set_xlabel('Reference Data Length')
# ax2.set_xscale('log')
ax3.set_ylabel('F1 Score')
ax3.set_ylim(0, 1.0)
ax3.legend(loc='center left', fontsize=11, bbox_to_anchor=(1.0, 0.5), fancybox=True, ncol=1)


# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
# plt.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

# Layout and show
# plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.17), fancybox=True, ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

