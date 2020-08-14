import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv('training.txt')
lines = file.plot.line(x='Epoch', y=['Train_loss', 'Valid_loss'])
plt.title('CNN learning curves for Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['test', 'validation'], loc='upper right')
#plt.show()
plt.savefig('Loss.png')