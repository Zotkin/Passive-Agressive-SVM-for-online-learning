# Passive-Agressive-SVM-for-online-learning
An implementation of Passive-Agressive version of SVM algorithm used for online learning. 

The implementation follows Scikit-Learn API.

This implementation allows to use Kernelized Version of SVM and different optimiation strategies. 

Avaliable Kernel Options: `linear`, `rbf`, `polynomial`
Avaliable Optimization Strategies:
- tau = loss / ||X||^2
- tau = min(c,loss / ||X||^2)
- tau = loss/(||X||^2 + (1/(2*C))) 
