from distillation import (train_knowledge_distillation,
                          cpu_device, nn_deep)
from Utility import test, train
from data_knowledge import train_loader, test_loader
import numpy as np
from quantum_student_model import LightNN, n_qubits, n_layers


nn_light = LightNN(n_layers, n_qubits).to(device=cpu_device)
print("Train quantum LIGHTER model...")
train(nn_light, train_loader, epochs=20, learning_rate=0.0001, seed=999, device=cpu_device)
test_accuracy_light = test(nn_light, test_loader, device=cpu_device)

new_nn_light = LightNN(n_layers, n_qubits).to(device=cpu_device)
myT = 2
# This is a training with different temperatures
'''T_temperatures = [2, 3, 4]
test_w_different_temperatures = np.zeros(len(T_temperatures))
for s, i in enumerate(range(len(T_temperatures))):
    print(f"Train quantum LIGHTER model with KNOWLEDGE-DISTILLATION...with T{i+1}")
    train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=20,
                                 learning_rate=0.0001, Temperature=T_temperatures[i], soft_target_loss_weight=0.3, ce_loss_weight=0.7,
                                 device=cpu_device, seed=s)
    test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device=cpu_device)
    test_w_different_temperatures[i] = test_accuracy_light_ce_and_kd
    print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")

print("Test quantum lighter model NOT DISTILLED: \n", test_accuracy_light)
print("")
print("Test quantum lighter models DISTILLED: \n", test_w_different_temperatures)'''

seed_list = np.arange(0, 5, 1)
test_w_different_seeds = np.zeros(len(seed_list))
for s in seed_list:
    print(f"Train quantum LIGHTER model with KNOWLEDGE-DISTILLATION...with T= ", myT)
    train_knowledge_distillation(teacher=nn_deep,
                                 student=new_nn_light,
                                 train_loader=train_loader,
                                 epochs=20,
                                 learning_rate=0.0001,
                                 Temperature=myT,
                                 soft_target_loss_weight=0.3,
                                 ce_loss_weight=0.7,
                                 device=cpu_device,
                                 seed=s)
    test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device=cpu_device)
    test_w_different_seeds[s] = test_accuracy_light_ce_and_kd
    print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")

print("Test quantum lighter model NOT DISTILLED:", test_accuracy_light)
print("Test accuracy mean of DISTILLED model: ", np.mean(test_w_different_seeds))