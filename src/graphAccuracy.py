import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv('training.txt')
lines = file.plot.line(x='Epoch', y=['Train_acc', 'Valid_acc'])
plt.title('CNN learning curves for Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['test', 'validation'], loc='lower right')
#plt.show()
plt.savefig('Accuracy.png')